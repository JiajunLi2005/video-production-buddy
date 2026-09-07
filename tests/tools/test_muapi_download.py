"""Exercise download policy without DNS or outbound network access."""
from __future__ import annotations

import io
import socket
from unittest.mock import Mock

import pytest

from tools.video import _muapi_download as download


def _address(ip="93.184.216.34"):
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    address = (ip, 443, 0, 0) if family == socket.AF_INET6 else (ip, 443)
    return family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", address


@pytest.mark.parametrize("ip", [
    "127.0.0.1", "10.0.0.1", "169.254.169.254", "192.168.1.1", "0.0.0.0",
    "::1", "fc00::1", "fe80::1", "::ffff:127.0.0.1", "224.0.0.1", "ff02::1",
])
def test_rejects_any_non_public_dns_answer(monkeypatch, ip):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_a, **_kw: [_address(), _address(ip)])
    with pytest.raises(ValueError, match="non-public"):
        download.resolve_output_url("https://media.example.test/video.mp4")


@pytest.mark.parametrize("url", [
    "http://media.example.test/v.mp4", "file:///tmp/video.mp4",
    "https://user:password@media.example.test/v.mp4", "https://media.example.test:444/v.mp4",
    "https://media.example.test/\nvideo.mp4",
])
def test_invalid_url_is_rejected_before_dns(monkeypatch, url):
    resolve = Mock(side_effect=AssertionError("DNS should not run"))
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    with pytest.raises(ValueError):
        download.resolve_output_url(url)
    resolve.assert_not_called()


def test_connection_pins_checked_address_and_verifies_original_tls_hostname(monkeypatch):
    raw = Mock()
    tls = Mock()
    context = Mock()
    context.wrap_socket.return_value = tls
    monkeypatch.setattr(socket, "socket", Mock(return_value=raw))
    monkeypatch.setattr(download.ssl, "create_default_context", lambda: context)
    monkeypatch.setattr(socket, "getaddrinfo", Mock(side_effect=AssertionError("must not resolve again")))
    connection = download._PinnedHTTPSConnection("media.example.test", _address(), 30)
    connection.connect()
    raw.connect.assert_called_once_with(("93.184.216.34", 443))
    context.wrap_socket.assert_called_once_with(raw, server_hostname="media.example.test")
    assert connection.host == "media.example.test"
    assert connection.sock is tls
    connection.close()


class _Response(io.BytesIO):
    def __init__(self, data, status, content_length):
        super().__init__(data)
        self.status = status
        self.content_length = content_length

    def getheader(self, name):
        assert name == "Content-Length"
        return self.content_length


def _mock_transport(monkeypatch, data=b"mp4 data", status=200, content_length=None):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_a, **_kw: [_address()])
    response = _Response(data, status, content_length)
    connection = Mock()
    connection.getresponse.return_value = response
    factory = Mock(return_value=connection)
    monkeypatch.setattr(download, "_PinnedHTTPSConnection", factory)
    return connection, factory


def test_streaming_download_omits_credentials_and_proxy_routing(monkeypatch, tmp_path):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1234")
    monkeypatch.setenv("MUAPI_API_KEY", "must-not-be-forwarded")
    connection, factory = _mock_transport(monkeypatch, content_length="8")
    output = tmp_path / "candidate.mp4"
    download.download_video("https://media.example.test/v.mp4?token=abc%2Fdef", output)
    assert output.read_bytes() == b"mp4 data"
    assert factory.call_args.args[:2] == ("media.example.test", _address())
    connection.request.assert_called_once_with(
        "GET", "/v.mp4?token=abc%2Fdef", headers={"Accept-Encoding": "identity"}
    )
    connection.close.assert_called_once()


@pytest.mark.parametrize("data,status,length,limit", [
    (b"redirect", 302, None, 100),
    (b"error", 500, None, 100),
    (b"", 200, None, 100),
    (b"", 200, "0", 100),
    (b"partial", 200, "100", 100),
    (b"too large", 200, None, 4),
    (b"too large", 200, "9", 4),
])
def test_download_rejects_redirects_empty_truncated_and_oversized_bodies(
    monkeypatch, tmp_path, data, status, length, limit,
):
    connection, factory = _mock_transport(monkeypatch, data, status, length)
    monkeypatch.setattr(download, "MAX_VIDEO_BYTES", limit)
    with pytest.raises(ValueError):
        download.download_video("https://media.example.test/out.mp4", tmp_path / "candidate.mp4")
    assert factory.call_count == 1
    connection.close.assert_called_once()


@pytest.mark.parametrize("phase", ["headers", "chunk_header", "chunk_trailer"])
def test_total_deadline_interrupts_drip_fed_http(monkeypatch, tmp_path, phase):
    import http.client
    import threading
    import time

    client, server = socket.socketpair()
    client.settimeout(5)
    server.settimeout(5)
    connection = http.client.HTTPConnection("media.example.test")
    connection.sock = client
    monkeypatch.setattr(connection, "connect", lambda: None)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_a, **_kw: [_address()])
    monkeypatch.setattr(download, "_PinnedHTTPSConnection", lambda *_a: connection)
    monkeypatch.setattr(download, "DOWNLOAD_TIMEOUT_SECONDS", 0.2)
    stop = threading.Event()

    def drip():
        try:
            server.recv(4096)
            if phase == "headers":
                server.sendall(b"HTTP/1.1 200 OK\r\nX-Drip: ")
            else:
                server.sendall(
                    b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
                )
                if phase == "chunk_trailer":
                    server.sendall(b"1\r\nx\r\n0\r\nX-Drip: ")
            for _ in range(50):
                if stop.wait(0.04):
                    break
                server.sendall(b"1")
        except OSError:
            pass  # The watchdog intentionally disconnects this peer.
        finally:
            server.close()

    peer = threading.Thread(target=drip, daemon=True)
    peer.start()
    start = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="timed out"):
            download.download_video("https://media.example.test/out.mp4", tmp_path / "candidate.mp4")
        assert time.monotonic() - start < 1.5
    finally:
        stop.set()
        connection.close()
        peer.join(timeout=2)
