"""Bounded, credential-free HTTPS downloads of MuAPI output media.

Resolve and validate every address, then connect to the chosen address directly
while retaining the original hostname for TLS verification and the Host header.
The stdlib transport deliberately does not use ambient proxies or netrc auth.
"""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
import threading
import time
from pathlib import Path
from urllib.parse import quote, urlsplit


MAX_VIDEO_BYTES = 500 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 300


def resolve_output_url(url: str) -> tuple[str, str, list[tuple]]:
    if any(ord(char) <= 32 for char in url):
        raise ValueError("MuAPI output URL contains whitespace/control characters")
    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").encode("idna").decode("ascii").lower()
    if parsed.scheme != "https" or not hostname:
        raise ValueError("MuAPI returned a non-HTTPS output URL")
    if parsed.username or parsed.password or parsed.port is not None:
        raise ValueError("MuAPI returned an output URL with credentials or a port")
    addresses = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
    if not addresses:
        raise ValueError("MuAPI output hostname has no addresses")
    for family, _kind, _proto, _name, address in addresses:
        ip = ipaddress.ip_address(address[0])
        if family not in {socket.AF_INET, socket.AF_INET6} or not ip.is_global or ip.is_multicast:
            raise ValueError("MuAPI output hostname resolves to a non-public address")
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    return hostname, quote(target, safe="/%:@!$&'()*+,;=-._~?"), addresses


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, hostname: str, address: tuple, timeout: float) -> None:
        self.address = address
        self.tls_context = ssl.create_default_context()
        super().__init__(hostname, timeout=timeout, context=self.tls_context)

    def connect(self) -> None:
        family, kind, proto, _name, address = self.address
        sock = socket.socket(family, kind, proto)
        try:
            sock.settimeout(self.timeout)
            sock.connect(address)
            self.sock = self.tls_context.wrap_socket(sock, server_hostname=self.host)
        except BaseException:
            sock.close()
            raise


def download_video(url: str, destination: Path) -> None:
    hostname, target, addresses = resolve_output_url(url)
    deadline = time.monotonic() + DOWNLOAD_TIMEOUT_SECONDS
    connection = None
    last_error = None
    for address in addresses:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("MuAPI output download timed out")
        candidate = _PinnedHTTPSConnection(hostname, address, min(30, remaining))
        try:
            candidate.connect()
        except OSError as exc:
            candidate.close()
            last_error = exc
            continue
        connection = candidate
        break
    if connection is None:
        raise OSError("Could not connect to MuAPI output host") from last_error

    # Socket timeouts limit inactivity, not total time. A peer can drip-feed
    # HTTP headers or chunk trailers forever. Keep the original socket even
    # when HTTPConnection releases it to a Connection: close response reader.
    active_socket = connection.sock
    expired = threading.Event()

    def abort_download() -> None:
        expired.set()
        try:
            active_socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # The response reader may already have closed the socket.

    watchdog = threading.Timer(max(0, deadline - time.monotonic()), abort_download)
    watchdog.daemon = True
    watchdog.start()
    try:
        connection.request("GET", target, headers={"Accept-Encoding": "identity"})
        with connection.getresponse() as response:
            if response.status != 200:
                raise ValueError(f"MuAPI output download returned HTTP {response.status}")
            content_length = response.getheader("Content-Length")
            expected_size = int(content_length) if content_length is not None else None
            if expected_size is not None and not 0 < expected_size <= MAX_VIDEO_BYTES:
                raise ValueError("MuAPI output size is empty or exceeds the download limit")
            total = 0
            with destination.open("wb") as handle:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("MuAPI output download timed out")
                    if connection.sock is not None:
                        connection.sock.settimeout(min(30, remaining))
                    chunk = response.read1(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_VIDEO_BYTES:
                        raise ValueError("MuAPI output exceeds the download limit")
                    handle.write(chunk)
            if total == 0 or (expected_size is not None and total != expected_size):
                raise ValueError("MuAPI output download is empty or truncated")
            if expired.is_set() or time.monotonic() >= deadline:
                raise TimeoutError("MuAPI output download timed out")
    except Exception as exc:
        if expired.is_set():
            raise TimeoutError("MuAPI output download timed out") from exc
        raise
    finally:
        watchdog.cancel()
        watchdog.join()
        connection.close()
