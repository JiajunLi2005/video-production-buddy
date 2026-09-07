from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tools.analysis.technical_qc import TechnicalQC


pytestmark = [pytest.mark.integration, pytest.mark.qa, pytest.mark.ffmpeg]


def _require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg/ffprobe are required for the media smoke test")


def test_ffmpeg_smoke_creates_probeable_video(tmp_path: Path) -> None:
    _require_ffmpeg()
    output = tmp_path / "smoke.mp4"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=24:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-shortest",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height",
            "-of",
            "default=nw=1",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert "codec_name=" in probe.stdout
    assert "width=320" in probe.stdout
    assert "height=180" in probe.stdout

    qc_result = TechnicalQC().execute(
        {
            "input_path": str(output),
            "expected": {
                "width": 320,
                "height": 180,
                "pixel_format": "yuv420p",
                "has_audio": True,
                "frame_rate": 24,
                "video_codec": "h264",
                "audio_codec": "aac",
                "audio_channels": 1,
                "audio_sample_rate": 44100,
                "max_file_size_mb": 10,
            },
        }
    )

    assert qc_result.success is True
    assert qc_result.data["status"] == "pass"
    assert (
        qc_result.data["metrics"]["audio_loudness"]["integrated_lufs"]
        is not None
    )


def test_technical_qc_detects_real_black_freeze_and_silence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_ffmpeg()
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "qc-smoke.mp4"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=24:duration=1",
            "-f",
            "lavfi",
            "-i",
            "color=black:size=320x180:rate=24:duration=2.5",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=24:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100:d=2.5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-filter_complex",
            (
                "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v];"
                "[3:a][4:a][5:a]concat=n=3:v=0:a=1[a]"
            ),
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(output),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )

    result = TechnicalQC().execute(
        {
            "input_path": str(output),
            "expected": {
                "width": 320,
                "height": 180,
                "pixel_format": "yuv420p",
                "has_audio": True,
            },
            "report_path": "projects/qc-smoke/artifacts/technical_qc.json",
        }
    )

    assert result.success is True
    assert result.data["status"] == "pass_with_warnings"
    issue_codes = {issue["code"] for issue in result.data["issues"]}
    assert {"black_segment", "freeze_segment", "silence_segment"} <= issue_codes
    assert result.data["metrics"]["black_segments"]
    assert result.data["metrics"]["freeze_segments"]
    assert result.data["metrics"]["silence_segments"]
    assert Path(result.data["report_path"]).is_file()
