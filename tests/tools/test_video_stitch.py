from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from tools.base_tool import ToolResult, ToolStatus
from tools.video.video_stitch import VideoStitch


class _RecordingVideoStitch(VideoStitch):
    def __init__(self) -> None:
        super().__init__()
        self.commands: list[list[str]] = []

    def run_command(self, cmd, *args, **kwargs):
        self.commands.append(list(cmd))

        class _Result:
            stdout = "{}"
            stderr = ""

        return _Result()


def test_video_stitch_rejects_unknown_media_profile_during_normalization():
    tool = VideoStitch()

    with pytest.raises(ValueError, match="Unknown profile"):
        tool._resolve_normalization_target(
            {"profile": "not-a-real-profile"},
            [{"width": 640, "height": 360, "fps": 24}],
        )


def test_video_stitch_idempotency_key_includes_output_and_render_parameters():
    tool = VideoStitch()
    base = {
        "operation": "stitch",
        "clips": ["a.mp4", "b.mp4"],
        "output_path": "out-a.mp4",
        "transition": "crossfade",
        "transition_duration": 0.5,
        "auto_normalize": True,
        "target_resolution": "640x360",
        "target_fps": 24,
        "codec": "libx264",
        "crf": 23,
        "preset": "medium",
        "profile": "generic_hd",
        "layout": "picture_in_picture",
        "pip_position": "bottom_right",
        "pip_scale": 0.3,
        "pip_margin": 10,
    }
    variants = [
        {"output_path": "out-b.mp4"},
        {"transition_duration": 1.0},
        {"auto_normalize": False},
        {"target_resolution": "1280x720"},
        {"target_fps": 30},
        {"codec": "libx265"},
        {"crf": 18},
        {"preset": "slow"},
        {"profile": "youtube_landscape"},
        {"pip_position": "top_left"},
        {"pip_scale": 0.4},
        {"pip_margin": 24},
    ]

    base_key = tool.idempotency_key(base)

    for variant in variants:
        assert tool.idempotency_key({**base, **variant}) != base_key


def test_video_stitch_dry_run_would_execute_reflects_dependency_status(monkeypatch):
    tool = VideoStitch()
    monkeypatch.setattr(tool, "get_status", lambda: ToolStatus.UNAVAILABLE)

    result = tool.dry_run({"operation": "stitch", "clips": []})

    assert result["status"] == ToolStatus.UNAVAILABLE.value
    assert result["would_execute"] is False


@pytest.mark.parametrize("output_kind", ["missing", "relative", "absolute"])
def test_video_stitch_requires_project_output_path_before_probe_or_stitch(
    output_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    clip_a = tmp_path / "a.mp4"
    clip_b = tmp_path / "b.mp4"
    clip_a.write_bytes(b"a")
    clip_b.write_bytes(b"b")
    tool = VideoStitch()
    probe_calls: list[str] = []
    stitch_calls: list[Path] = []

    def fake_probe(path: str) -> dict[str, object]:
        probe_calls.append(path)
        return {
            "width": 640,
            "height": 360,
            "fps": 24,
            "duration": 1.0,
            "video_codec": "h264",
            "audio_codec": "aac",
            "sample_rate": 44100,
        }

    def fake_stitch_cut(
        clips: list[str],
        output_path: Path,
        temp_dir: Path,
        temp_files: list[Path],
    ) -> dict[str, object]:
        stitch_calls.append(output_path)
        output_path.write_bytes(b"stitched")
        return {"method": "concat_demuxer"}

    monkeypatch.setattr(tool, "_probe_clip", fake_probe)
    monkeypatch.setattr(tool, "_stitch_cut", fake_stitch_cut)
    inputs: dict[str, object] = {
        "operation": "stitch",
        "clips": [str(clip_a), str(clip_b)],
    }
    if output_kind == "relative":
        inputs["output_path"] = "stitched.mp4"
        forbidden_output = tmp_path / "stitched.mp4"
    elif output_kind == "absolute":
        forbidden_output = tmp_path / "stitched.mp4"
        inputs["output_path"] = str(forbidden_output)
    else:
        forbidden_output = tmp_path / "stitched_output.mp4"

    result = tool.execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert probe_calls == []
    assert stitch_calls == []
    assert not forbidden_output.exists()


@pytest.mark.parametrize("output_kind", ["missing", "relative", "absolute"])
def test_video_stitch_preview_requires_project_output_path_before_stitch(
    output_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    clip_a = tmp_path / "a.mp4"
    clip_b = tmp_path / "b.mp4"
    clip_a.write_bytes(b"a")
    clip_b.write_bytes(b"b")
    tool = VideoStitch()
    stitch_inputs: list[dict[str, object]] = []

    def fake_stitch(inputs: dict[str, object]) -> ToolResult:
        stitch_inputs.append(inputs)
        output_path = Path(str(inputs["output_path"]))
        output_path.write_bytes(b"preview")
        return ToolResult(success=True, data={"operation": "stitch"})

    monkeypatch.setattr(tool, "_stitch", fake_stitch)
    inputs: dict[str, object] = {
        "operation": "preview_stitch",
        "clips": [str(clip_a), str(clip_b)],
    }
    if output_kind == "relative":
        inputs["output_path"] = "preview.mp4"
        forbidden_output = tmp_path / "preview.mp4"
    elif output_kind == "absolute":
        forbidden_output = tmp_path / "preview.mp4"
        inputs["output_path"] = str(forbidden_output)
    else:
        forbidden_output = tmp_path / "stitch_preview.mp4"

    result = tool.execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert stitch_inputs == []
    assert not forbidden_output.exists()


@pytest.mark.parametrize("output_kind", ["missing", "relative", "absolute"])
def test_video_stitch_spatial_requires_project_output_path_before_render(
    output_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    clip_a = tmp_path / "a.mp4"
    clip_b = tmp_path / "b.mp4"
    clip_a.write_bytes(b"a")
    clip_b.write_bytes(b"b")
    tool = VideoStitch()
    ensure_calls: list[list[str]] = []
    render_calls: list[Path] = []
    probe_calls: list[str] = []

    def fake_ensure(
        clips: list[str],
        temp_dir: Path,
        temp_files: list[Path],
    ) -> list[str]:
        ensure_calls.append(list(clips))
        return list(clips)

    def fake_side_by_side(
        clips: list[str],
        output_path: Path,
        codec: str,
        crf: int,
    ) -> None:
        render_calls.append(output_path)
        output_path.write_bytes(b"spatial")

    def fake_probe(path: str) -> dict[str, object]:
        probe_calls.append(path)
        return {"duration": 1.0}

    monkeypatch.setattr(tool, "_ensure_audio_for_clips", fake_ensure)
    monkeypatch.setattr(tool, "_spatial_side_by_side", fake_side_by_side)
    monkeypatch.setattr(tool, "_probe_clip", fake_probe)
    inputs: dict[str, object] = {
        "operation": "spatial",
        "clips": [str(clip_a), str(clip_b)],
        "layout": "side_by_side",
    }
    if output_kind == "relative":
        inputs["output_path"] = "spatial.mp4"
        forbidden_output = tmp_path / "spatial.mp4"
    elif output_kind == "absolute":
        forbidden_output = tmp_path / "spatial.mp4"
        inputs["output_path"] = str(forbidden_output)
    else:
        forbidden_output = tmp_path / "spatial_output.mp4"

    result = tool.execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert ensure_calls == []
    assert render_calls == []
    assert probe_calls == []
    assert not forbidden_output.exists()


def test_video_stitch_success_payload_includes_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    clip_a = tmp_path / "a.mp4"
    clip_b = tmp_path / "b.mp4"
    clip_a.write_bytes(b"a")
    clip_b.write_bytes(b"b")
    output_path = Path("projects/demo/renders/stitched.mp4")
    tool = VideoStitch()

    def fake_probe(path: str) -> dict[str, object]:
        return {
            "width": 640,
            "height": 360,
            "fps": 24,
            "duration": 2.0,
            "video_codec": "h264",
            "audio_codec": "aac",
            "sample_rate": 44100,
        }

    def fake_stitch_cut(
        clips: list[str],
        output_path: Path,
        temp_dir: Path,
        temp_files: list[Path],
    ) -> dict[str, object]:
        output_path.write_bytes(b"stitched")
        return {"method": "concat_demuxer"}

    monkeypatch.setattr(tool, "_probe_clip", fake_probe)
    monkeypatch.setattr(tool, "_stitch_cut", fake_stitch_cut)

    result = tool.execute(
        {
            "operation": "stitch",
            "clips": [str(clip_a), str(clip_b)],
            "output_path": str(output_path),
        }
    )

    assert result.success is True
    assert result.data is not None
    assert result.data["output"] == output_path.as_posix()
    assert result.data["output_path"] == output_path.as_posix()
    assert result.artifacts == [output_path.as_posix()]
    assert {
        "operation",
        "clip_count",
        "transition",
        "transition_duration",
        "auto_normalized",
        "output",
        "output_path",
        "duration",
        "file_size_bytes",
        "method",
        "layout",
    } <= set(VideoStitch.output_schema["properties"])
    jsonschema.validate(instance=result.data, schema=VideoStitch.output_schema)


def test_video_stitch_spatial_success_payload_includes_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    clip_a = tmp_path / "a.mp4"
    clip_b = tmp_path / "b.mp4"
    clip_a.write_bytes(b"a")
    clip_b.write_bytes(b"b")
    output_path = Path("projects/demo/renders/spatial.mp4")
    tool = VideoStitch()

    def fake_ensure(
        clips: list[str],
        temp_dir: Path,
        temp_files: list[Path],
    ) -> list[str]:
        return list(clips)

    def fake_side_by_side(
        clips: list[str],
        output_path: Path,
        codec: str,
        crf: int,
    ) -> None:
        output_path.write_bytes(b"spatial")

    def fake_probe(path: str) -> dict[str, object]:
        return {"duration": 2.0}

    monkeypatch.setattr(tool, "_ensure_audio_for_clips", fake_ensure)
    monkeypatch.setattr(tool, "_spatial_side_by_side", fake_side_by_side)
    monkeypatch.setattr(tool, "_probe_clip", fake_probe)

    result = tool.execute(
        {
            "operation": "spatial",
            "clips": [str(clip_a), str(clip_b)],
            "layout": "side_by_side",
            "output_path": str(output_path),
        }
    )

    assert result.success is True
    assert result.data is not None
    assert result.data["output"] == output_path.as_posix()
    assert result.data["output_path"] == output_path.as_posix()
    assert result.artifacts == [output_path.as_posix()]
    jsonschema.validate(instance=result.data, schema=VideoStitch.output_schema)


def test_video_stitch_validate_success_payload_matches_output_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    clip_a = tmp_path / "a.mp4"
    clip_b = tmp_path / "b.mp4"
    clip_a.write_bytes(b"a")
    clip_b.write_bytes(b"b")
    tool = VideoStitch()

    def fake_probe(path: str) -> dict[str, object]:
        return {
            "path": path,
            "width": 640,
            "height": 360,
            "fps": 24,
            "duration": 1.5,
            "video_codec": "h264",
            "pixel_format": "yuv420p",
            "audio_codec": "aac",
            "sample_rate": 44100,
            "audio_channels": 2,
        }

    monkeypatch.setattr(tool, "_probe_clip", fake_probe)

    result = tool.execute(
        {
            "operation": "validate",
            "clips": [str(clip_a), str(clip_b)],
        }
    )

    assert result.success is True
    assert result.data is not None
    assert result.data["operation"] == "validate"
    assert result.data["compatible"] is True
    assert result.data["total_duration"] == 3.0
    jsonschema.validate(instance=result.data, schema=VideoStitch.output_schema)


@pytest.mark.parametrize("operation", ["stitch", "preview_stitch", "spatial"])
def test_video_stitch_schemas_require_two_clips_for_composition_operations(
    operation: str,
):
    canonical_schema = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "schemas"
            / "tools"
            / "video_stitch.schema.json"
        ).read_text(encoding="utf-8")
    )
    instance = {"operation": operation, "clips": ["one.mp4"]}
    if operation == "spatial":
        instance["layout"] = "side_by_side"

    for schema in (canonical_schema, VideoStitch.input_schema):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=instance, schema=schema)


def test_stitch_cut_escapes_single_quotes_in_concat_list_paths(tmp_path: Path):
    clip = tmp_path / "clip's intro.mp4"
    clip.write_bytes(b"placeholder")
    other = tmp_path / "other.mp4"
    other.write_bytes(b"placeholder")
    output = tmp_path / "out.mp4"
    temp_dir = tmp_path / ".stitch_tmp"
    temp_dir.mkdir()
    temp_files: list[Path] = []
    tool = _RecordingVideoStitch()

    tool._stitch_cut([str(clip), str(other)], output, temp_dir, temp_files)

    concat_list = temp_dir / "concat_list.txt"
    body = concat_list.read_text(encoding="utf-8")
    assert "clip'\\''s intro.mp4" in body
