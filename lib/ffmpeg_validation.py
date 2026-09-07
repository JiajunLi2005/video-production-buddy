"""Decode failure handling shared by media acceptance and technical QC."""

from __future__ import annotations

import re


# Some FFmpeg versions report decoder-thread errors but exit with status 0.
# Include severity labels so callers can reject those errors explicitly.
STRICT_DECODE_ARGS = ["-loglevel", "level+info", "-xerror", "-err_detect", "explode"]


def require_clean_decode(stderr: str) -> None:
    """Reject failed/concealed decoding even after a successful process exit."""
    for line in stderr.splitlines():
        line = line.strip()
        # Severity labels precede the message. A quoted filename or metadata
        # value containing "[error]" is not a decoder failure.
        error_level = re.match(
            r"^(?:\[[^\]]+\]\s+)?\[(?:error|fatal|panic)\](?:\s|$)", line, re.I
        )
        message = re.sub(r"^(?:\[[^\]]+\]\s*)+", "", line)
        concealed = re.match(r"corrupt decoded frame|concealing .* errors in", message, re.I)
        if error_level or concealed:
            raise ValueError(f"FFmpeg decoding failed: {line.strip()[:500]}")
