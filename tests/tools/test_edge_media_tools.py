from __future__ import annotations

import builtins
import math
from pathlib import Path
import types
from typing import Any

import jsonschema
import pytest

from tools.audio.audio_enhance import AudioEnhance
from tools.avatar.lip_sync import LipSync
from tools.avatar.talking_head import TalkingHead
from tools.capture.cap_recorder import CapRecorder
from tools.enhancement.bg_remove import BgRemove
from tools.enhancement.color_grade import ColorGrade
from tools.enhancement.eye_enhance import EyeEnhance
from tools.enhancement.face_enhance import FaceEnhance
from tools.enhancement.face_restore import FaceRestore
from tools.enhancement.upscale import Upscale
from tools.base_tool import ToolResult, ToolStatus
from tools.subtitle.subtitle_gen import SubtitleGen


def _project_avatar_output(name: str) -> str:
    return f"projects/test-avatar/assets/video/{name}"


def _assert_output_payload_matches_schema(
    tool: Any,
    payload: dict[str, object],
    expected_properties: set[str],
) -> None:
    output_properties = tool.output_schema["properties"]
    assert expected_properties <= set(output_properties)
    jsonschema.validate(instance=payload, schema=tool.output_schema)


def test_cap_pick_latest_accepts_exact_output_file_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recordings_root = tmp_path / "cap"
    source = recordings_root / "session-1" / "output" / "result.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"cap-recording")
    requested_output = Path("projects/demo/renders/picked.mp4")

    monkeypatch.setattr(
        "tools.capture.cap_recorder._find_cap_recordings_dir",
        lambda: recordings_root,
    )

    result = CapRecorder().execute(
        {"operation": "pick_latest", "output_dir": str(requested_output)}
    )

    assert result.success
    assert result.data["output_path"] == requested_output.as_posix()
    assert (tmp_path / requested_output).read_bytes() == b"cap-recording"


def test_cap_recorder_idempotency_key_includes_pick_latest_output_dir():
    tool = CapRecorder()
    base = {"operation": "pick_latest", "output_dir": "projects/demo/renders"}

    assert tool.idempotency_key(base) != tool.idempotency_key(
        {**base, "output_dir": "projects/demo/assets/raw"}
    )


def test_talking_head_schema_does_not_offer_unimplemented_musetalk():
    model_schema = TalkingHead.input_schema["properties"]["model"]

    assert "musetalk" not in model_schema["enum"]


def test_talking_head_success_payload_matches_output_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    image_path = tmp_path / "face.png"
    audio_path = tmp_path / "voice.wav"
    image_path.write_bytes(b"image")
    audio_path.write_bytes(b"audio")
    sadtalker_dir = tmp_path / "sadtalker"
    sadtalker_dir.mkdir()
    (sadtalker_dir / "inference.py").write_text("# fake sadtalker\n", encoding="utf-8")
    monkeypatch.setenv("SADTALKER_PATH", str(sadtalker_dir))
    output_path = _project_avatar_output("talking-head.mp4")

    def fake_run_command(self: TalkingHead, cmd: list[str], **kwargs: object) -> None:
        result_dir = Path(cmd[cmd.index("--result_dir") + 1])
        generated = result_dir / "session" / "generated.mp4"
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_bytes(b"talking-head")

    monkeypatch.setattr(TalkingHead, "run_command", fake_run_command)

    result = TalkingHead().execute(
        {
            "image_path": str(image_path),
            "audio_path": str(audio_path),
            "output_path": output_path,
            "expression_scale": 1.25,
            "still_mode": True,
            "preprocess": "full",
        }
    )

    assert result.success is True
    assert result.data is not None
    assert result.data["output_path"] == output_path
    assert result.artifacts == [output_path]
    assert (tmp_path / output_path).read_bytes() == b"talking-head"
    _assert_output_payload_matches_schema(
        TalkingHead,
        result.data,
        {
            "model",
            "image",
            "audio",
            "output",
            "output_path",
            "expression_scale",
            "still_mode",
            "preprocess",
            "format",
        },
    )


@pytest.mark.parametrize("output_kind", ["missing", "relative", "absolute"])
def test_talking_head_requires_project_output_path_before_generation(
    output_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    image_path = tmp_path / "face.png"
    audio_path = tmp_path / "voice.wav"
    image_path.write_bytes(b"image")
    audio_path.write_bytes(b"audio")
    generation_calls: list[Path] = []

    def fake_run(
        self: TalkingHead,
        inputs: dict[str, Any],
        image_path: Path,
        audio_path: Path,
        output_path: Path,
    ):
        generation_calls.append(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"video")
        from tools.base_tool import ToolResult

        return ToolResult(success=True, artifacts=[str(output_path)])

    monkeypatch.setattr(TalkingHead, "_run_sadtalker", fake_run)
    inputs: dict[str, object] = {
        "image_path": str(image_path),
        "audio_path": str(audio_path),
    }
    if output_kind == "relative":
        inputs["output_path"] = "talking.mp4"
        forbidden_output = tmp_path / "talking.mp4"
    elif output_kind == "absolute":
        forbidden_output = tmp_path / "talking.mp4"
        inputs["output_path"] = str(forbidden_output)
    else:
        forbidden_output = image_path.with_stem(f"{image_path.stem}_talking").with_suffix(".mp4")

    result = TalkingHead().execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert generation_calls == []
    assert not forbidden_output.exists()


@pytest.mark.parametrize("output_kind", ["missing", "relative", "absolute"])
def test_lip_sync_requires_project_output_path_before_checkpoint_lookup(
    output_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "speaker.mp4"
    audio_path = tmp_path / "voice.wav"
    video_path.write_bytes(b"video")
    audio_path.write_bytes(b"audio")
    resolve_calls: list[str] = []

    monkeypatch.setattr(LipSync, "get_status", lambda self: ToolStatus.AVAILABLE)

    def fake_resolve(self: LipSync):
        resolve_calls.append("resolve")
        return tmp_path / "wav2lip"

    monkeypatch.setattr(LipSync, "_resolve_wav2lip_dir", fake_resolve)
    inputs: dict[str, object] = {
        "video_path": str(video_path),
        "audio_path": str(audio_path),
    }
    if output_kind == "relative":
        inputs["output_path"] = "lipsync.mp4"
        forbidden_output = tmp_path / "lipsync.mp4"
    elif output_kind == "absolute":
        forbidden_output = tmp_path / "lipsync.mp4"
        inputs["output_path"] = str(forbidden_output)
    else:
        forbidden_output = video_path.with_stem(f"{video_path.stem}_lipsync")

    result = LipSync().execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert resolve_calls == []
    assert not forbidden_output.exists()


def test_lip_sync_success_payload_includes_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "speaker.mp4"
    audio_path = tmp_path / "voice.wav"
    video_path.write_bytes(b"video")
    audio_path.write_bytes(b"audio")
    wav2lip_dir = tmp_path / "wav2lip"
    checkpoint = wav2lip_dir / "checkpoints" / "wav2lip.pth"
    inference_script = wav2lip_dir / "inference.py"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    inference_script.write_text("# fake inference\n", encoding="utf-8")
    output_path = _project_avatar_output("lipsync.mp4")
    tool = LipSync()

    monkeypatch.setattr(tool, "get_status", lambda: ToolStatus.AVAILABLE)
    monkeypatch.setattr(tool, "_resolve_wav2lip_dir", lambda: wav2lip_dir)

    def fake_run_command(cmd, *args, **kwargs):
        rendered = Path(cmd[cmd.index("--outfile") + 1])
        rendered.parent.mkdir(parents=True, exist_ok=True)
        rendered.write_bytes(b"lipsync")

    monkeypatch.setattr(tool, "run_command", fake_run_command)

    result = tool.execute(
        {
            "video_path": str(video_path),
            "audio_path": str(audio_path),
            "output_path": output_path,
        }
    )

    assert result.success is True
    assert result.data is not None
    assert result.data["output"] == output_path
    assert result.data["output_path"] == output_path
    assert result.artifacts == [output_path]
    assert (tmp_path / output_path).read_bytes() == b"lipsync"
    _assert_output_payload_matches_schema(
        LipSync,
        result.data,
        {
            "video_input",
            "audio_input",
            "output",
            "output_path",
            "model",
            "face_padding",
            "resize_factor",
        },
    )


@pytest.mark.parametrize(
    ("tool_cls", "instance"),
    [
        (TalkingHead, {"image_path": "face.png", "audio_path": "voice.wav"}),
        (LipSync, {"video_path": "speaker.mp4", "audio_path": "voice.wav"}),
    ],
)
def test_avatar_generation_schemas_require_output_path(
    tool_cls: type[TalkingHead] | type[LipSync],
    instance: dict[str, object],
) -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=instance, schema=tool_cls.input_schema)


@pytest.mark.parametrize(
    ("tool", "instance"),
    [
        (BgRemove(), {"input_path": "subject.png"}),
        (ColorGrade(), {"input_path": "scene.mp4"}),
        (EyeEnhance(), {"input_path": "speaker.mp4"}),
        (FaceEnhance(), {"input_path": "speaker.mp4"}),
        (FaceRestore(), {"input_path": "face.png"}),
        (Upscale(), {"input_path": "small.png"}),
        (AudioEnhance(), {"input_path": "voice.wav"}),
    ],
)
def test_enhancement_schemas_require_output_path(
    tool: Any,
    instance: dict[str, object],
) -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=instance, schema=tool.input_schema)


def test_lip_sync_rejects_unknown_model_before_checkpoint_lookup(tmp_path, monkeypatch):
    wav2lip_dir = tmp_path / "wav2lip"
    wav2lip_dir.mkdir()
    video_path = tmp_path / "source.mp4"
    audio_path = tmp_path / "voice.wav"
    video_path.write_bytes(b"video")
    audio_path.write_bytes(b"audio")
    monkeypatch.setenv("WAV2LIP_PATH", str(wav2lip_dir))

    result = LipSync().execute(
        {
            "video_path": str(video_path),
            "audio_path": str(audio_path),
            "model": "not-a-real-model",
            "output_path": _project_avatar_output("lipsync.mp4"),
        }
    )

    assert not result.success
    assert "Unknown model" in (result.error or "")


def test_lip_sync_status_respects_declared_ffmpeg_dependency(tmp_path, monkeypatch):
    wav2lip_dir = tmp_path / "wav2lip"
    wav2lip_dir.mkdir()
    monkeypatch.setenv("WAV2LIP_PATH", str(wav2lip_dir))
    monkeypatch.setattr("tools.base_tool.shutil.which", lambda command: None)

    assert LipSync().get_status() == ToolStatus.UNAVAILABLE


def test_face_restore_status_requires_cv2_runtime_dependency(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"gfpgan", "torch"}:
            return types.SimpleNamespace()
        if name == "cv2":
            raise ImportError("cv2 missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert FaceRestore().get_status() == ToolStatus.UNAVAILABLE


@pytest.mark.parametrize(
    ("tool", "present_module", "missing_module"),
    [
        (BgRemove(), "rembg", "PIL"),
        (Upscale(), "realesrgan", "torch"),
    ],
)
def test_enhancement_status_respects_all_declared_dependencies(
    tool: Any,
    present_module: str,
    missing_module: str,
    monkeypatch: pytest.MonkeyPatch,
):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == present_module:
            return types.SimpleNamespace()
        if name == missing_module:
            raise ImportError(f"{missing_module} missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert tool.get_status() == ToolStatus.UNAVAILABLE


@pytest.mark.parametrize(
    ("tool", "base", "variant"),
    [
        (
            SubtitleGen(),
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "Cloud code",
                        "words": [
                            {"word": "Cloud", "start": 0.0, "end": 0.4},
                            {"word": "code", "start": 0.4, "end": 1.0},
                        ],
                    }
                ],
                "format": "srt",
                "max_words_per_cue": 8,
            },
            {"output_path": "captions-a.srt"},
        ),
        (
            SubtitleGen(),
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "Cloud code",
                        "words": [
                            {"word": "Cloud", "start": 0.0, "end": 0.4},
                            {"word": "code", "start": 0.4, "end": 1.0},
                        ],
                    }
                ],
                "format": "srt",
                "max_words_per_cue": 8,
            },
            {"corrections": {"cloud": "Claude"}},
        ),
        (
            FaceEnhance(),
            {"input_path": "speaker.mp4", "preset": "sharpen"},
            {"codec": "libx265"},
        ),
        (
            ColorGrade(),
            {"input_path": "scene.mp4", "profile": "neutral", "intensity": 1.0},
            {"custom_vf": "eq=brightness=0.1"},
        ),
        (
            AudioEnhance(),
            {"input_path": "voice.wav", "preset": "clean_speech"},
            {"audio_bitrate": "96k"},
        ),
        (
            FaceRestore(),
            {"input_path": "face.png", "model": "CodeFormer", "fidelity": 0.5, "upscale": 2},
            {"bg_upsampler": True},
        ),
        (
            FaceRestore(),
            {"input_path": "face.png", "model": "CodeFormer", "fidelity": 0.5, "upscale": 2},
            {"output_path": "restored-a.png"},
        ),
        (
            BgRemove(),
            {"input_path": "subject.png", "model": "u2net", "alpha_matting": False},
            {"output_path": "subject-a.png"},
        ),
        (
            Upscale(),
            {
                "input_path": "small.png",
                "scale": 4,
                "model": "RealESRGAN_x4plus",
                "face_enhance": False,
                "denoise_strength": 0.5,
            },
            {"output_path": "large-a.png"},
        ),
        (
            TalkingHead(),
            {
                "image_path": "face.png",
                "audio_path": "voice.wav",
                "model": "sadtalker",
                "expression_scale": 1.0,
                "still_mode": False,
            },
            {"output_path": "talking-head-a.mp4"},
        ),
        (
            TalkingHead(),
            {
                "image_path": "face.png",
                "audio_path": "voice.wav",
                "model": "sadtalker",
                "expression_scale": 1.0,
                "still_mode": False,
            },
            {"preprocess": "full"},
        ),
        (
            LipSync(),
            {
                "video_path": "speaker.mp4",
                "audio_path": "voice.wav",
                "model": "wav2lip",
                "face_padding": [0, 10, 0, 0],
                "resize_factor": 1,
            },
            {"output_path": "lipsync-a.mp4"},
        ),
    ],
)
def test_edge_media_idempotency_keys_include_output_shaping_inputs(
    tool: Any,
    base: dict[str, object],
    variant: dict[str, object],
):
    assert tool.idempotency_key(base) != tool.idempotency_key({**base, **variant})


def test_subtitle_timestamps_carry_when_rounding_to_next_second():
    assert SubtitleGen._ts_srt(1.9996) == "00:00:02,000"
    assert SubtitleGen._ts_vtt(1.9996) == "00:00:02.000"


@pytest.mark.parametrize(
    "output_path",
    [
        None,
        "subtitles.srt",
        "/tmp/subtitles.srt",
    ],
)
def test_subtitle_gen_requires_project_output_path_before_writing(
    output_path: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    inputs: dict[str, object] = {
        "segments": [{"text": "Hello", "start": 0.0, "end": 1.0}],
        "format": "srt",
    }
    if output_path is not None:
        inputs["output_path"] = output_path

    result = SubtitleGen().execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert not (tmp_path / "subtitles.srt").exists()


def test_subtitle_gen_schema_requires_output_path() -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            instance={"segments": [{"text": "Hello", "start": 0.0, "end": 1.0}]},
            schema=SubtitleGen.input_schema,
        )


def test_subtitle_gen_success_payload_includes_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    output_path = "projects/demo/assets/subtitles/captions.srt"

    result = SubtitleGen().execute(
        {
            "segments": [{"text": "Hello", "start": 0.0, "end": 1.0}],
            "format": "srt",
            "output_path": output_path,
        }
    )

    assert result.success is True
    assert result.data["output_path"] == output_path
    assert result.artifacts == [output_path]
    assert (tmp_path / output_path).exists()


def test_subtitle_gen_success_payload_matches_output_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    output_path = "projects/demo/assets/subtitles/captions-schema.srt"

    output_properties = SubtitleGen.output_schema["properties"]
    assert {
        "format",
        "cue_count",
        "output",
        "output_path",
    } <= set(output_properties)

    result = SubtitleGen().execute(
        {
            "segments": [{"text": "Hello", "start": 0.0, "end": 1.0}],
            "format": "srt",
            "output_path": output_path,
        }
    )

    assert result.success is True
    assert result.data == {
        "format": "srt",
        "cue_count": 1,
        "output": output_path,
        "output_path": output_path,
    }
    assert result.artifacts == [output_path]
    assert (tmp_path / output_path).exists()
    jsonschema.validate(instance=result.data, schema=SubtitleGen.output_schema)


def test_subtitle_gen_malformed_word_timing_returns_tool_error(tmp_path: Path) -> None:
    try:
        result = SubtitleGen().execute(
            {
                "segments": [
                    {
                        "text": "broken timing",
                        "start": 0.0,
                        "end": 1.0,
                        "words": [{"word": "broken"}],
                    }
                ],
                "output_path": str(tmp_path / "broken.srt"),
            }
        )
    except Exception as exc:  # pragma: no cover - documents current bug shape
        pytest.fail(f"execute raised instead of returning ToolResult: {exc}")

    assert result.success is False
    assert "timestamp" in (result.error or "").lower()


def test_subtitle_gen_rejects_non_finite_caption_json_before_writing(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "captions.caption.json"

    result = SubtitleGen().execute(
        {
            "segments": [
                {
                    "text": "bad timing",
                    "start": 0.0,
                    "end": math.nan,
                }
            ],
            "format": "json",
            "output_path": str(output_path),
        }
    )

    assert result.success is False
    assert "strict JSON" in (result.error or "")
    assert not output_path.exists()


def test_color_grade_zero_intensity_uses_noop_filter():
    assert ColorGrade()._build_filter({"profile": "cinematic_warm", "intensity": 0.0}) == "null"


def test_color_grade_missing_lut_path_returns_tool_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "scene.mp4"
    input_path.write_bytes(b"video")
    output_path = tmp_path / "projects" / "demo" / "renders" / "graded.mp4"
    monkeypatch.setattr(ColorGrade, "run_command", lambda *args, **kwargs: output_path.write_bytes(b"graded"))

    result = ColorGrade().execute(
        {
            "input_path": str(input_path),
            "output_path": "projects/demo/renders/graded.mp4",
            "lut_path": str(tmp_path / "missing.cube"),
        }
    )

    assert result.success is False
    assert "LUT not found" in (result.error or "")


@pytest.mark.parametrize("output_kind", ["missing", "relative", "absolute"])
def test_color_grade_requires_project_output_path_before_ffmpeg(
    output_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "scene.mp4"
    input_path.write_bytes(b"video")
    commands: list[list[str]] = []

    def fake_run_command(self: ColorGrade, cmd: list[str], **kwargs: object) -> None:
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"graded")

    monkeypatch.setattr(ColorGrade, "run_command", fake_run_command)
    inputs: dict[str, object] = {
        "input_path": str(input_path),
        "profile": "neutral",
    }
    if output_kind == "relative":
        inputs["output_path"] = "graded.mp4"
        forbidden_output = tmp_path / "graded.mp4"
    elif output_kind == "absolute":
        forbidden_output = tmp_path / "graded.mp4"
        inputs["output_path"] = str(forbidden_output)
    else:
        forbidden_output = tmp_path / "scene_graded.mp4"

    result = ColorGrade().execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert commands == []
    assert not forbidden_output.exists()


def test_color_grade_escapes_single_quotes_in_lut_filter_path(tmp_path: Path):
    lut_dir = tmp_path / "look's"
    lut_dir.mkdir()
    lut_path = lut_dir / "grade.cube"
    lut_path.write_text("LUT", encoding="utf-8")

    vf = ColorGrade()._build_filter({"lut_path": str(lut_path)})

    assert "look\\'s" in vf


def test_face_enhance_rejects_unknown_preset_in_multi_preset_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "speaker.mp4"
    input_path.write_bytes(b"video")
    output_path = tmp_path / "projects" / "demo" / "renders" / "enhanced.mp4"
    monkeypatch.setattr(FaceEnhance, "run_command", lambda *args, **kwargs: output_path.write_bytes(b"enhanced"))

    result = FaceEnhance().execute(
        {
            "input_path": str(input_path),
            "output_path": "projects/demo/renders/enhanced.mp4",
            "presets": ["sharpen", "not-a-real-preset"],
        }
    )

    assert result.success is False
    assert "Unknown preset" in (result.error or "")


@pytest.mark.parametrize("output_kind", ["missing", "relative", "absolute"])
def test_face_enhance_requires_project_output_path_before_ffmpeg(
    output_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "speaker.mp4"
    input_path.write_bytes(b"video")
    commands: list[list[str]] = []

    def fake_run_command(self: FaceEnhance, cmd: list[str], **kwargs: object) -> None:
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"enhanced")

    monkeypatch.setattr(FaceEnhance, "run_command", fake_run_command)
    inputs: dict[str, object] = {
        "input_path": str(input_path),
        "preset": "sharpen",
    }
    if output_kind == "relative":
        inputs["output_path"] = "enhanced.mp4"
        forbidden_output = tmp_path / "enhanced.mp4"
    elif output_kind == "absolute":
        forbidden_output = tmp_path / "enhanced.mp4"
        inputs["output_path"] = str(forbidden_output)
    else:
        forbidden_output = tmp_path / "speaker_enhanced.mp4"

    result = FaceEnhance().execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert commands == []
    assert not forbidden_output.exists()


@pytest.mark.parametrize("output_kind", ["missing", "relative", "absolute"])
def test_bg_remove_requires_project_output_path_before_rembg_call(
    output_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "subject.png"
    input_path.write_bytes(b"image")
    original_import = builtins.__import__
    remove_calls: list[dict[str, object]] = []

    class FakeImage:
        size = (1, 1)

        def save(self, path: str) -> None:
            Path(path).write_bytes(b"image")

    fake_image_module = types.SimpleNamespace(open=lambda path: FakeImage())

    def fake_remove(image: object, **kwargs: object) -> FakeImage:
        remove_calls.append(kwargs)
        return FakeImage()

    def fake_import(name, *args, **kwargs):
        if name == "rembg":
            return types.SimpleNamespace(remove=fake_remove)
        if name == "PIL":
            return types.SimpleNamespace(Image=fake_image_module)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    inputs: dict[str, object] = {
        "input_path": str(input_path),
        "model": "u2net",
    }
    if output_kind == "relative":
        inputs["output_path"] = "subject-nobg.png"
        forbidden_output = tmp_path / "subject-nobg.png"
    elif output_kind == "absolute":
        forbidden_output = tmp_path / "outside.png"
        inputs["output_path"] = str(forbidden_output)
    else:
        forbidden_output = tmp_path / "subject_nobg.png"

    result = BgRemove().execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert remove_calls == []
    assert not forbidden_output.exists()


def test_bg_remove_rejects_unknown_model_before_rembg_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "subject.png"
    input_path.write_bytes(b"image")
    original_import = builtins.__import__

    class FakeImage:
        size = (1, 1)

        def save(self, path: str) -> None:
            Path(path).write_bytes(b"image")

    fake_image_module = types.SimpleNamespace(open=lambda path: FakeImage())

    def fake_import(name, *args, **kwargs):
        if name == "rembg":
            return types.SimpleNamespace(remove=lambda image, **kwargs: FakeImage())
        if name == "PIL":
            return types.SimpleNamespace(Image=fake_image_module)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = BgRemove().execute(
        {
            "input_path": str(input_path),
            "output_path": "projects/demo/assets/images/subject-nobg.png",
            "model": "not-a-real-model",
        }
    )

    assert result.success is False
    assert "Unknown model" in (result.error or "")


def test_bg_remove_success_payload_includes_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "subject.png"
    input_path.write_bytes(b"image")
    output_path = "projects/demo/assets/images/subject-nobg.png"
    original_import = builtins.__import__

    class FakeImage:
        size = (1, 1)

        def save(self, path: str) -> None:
            Path(path).write_bytes(b"image")

    fake_image_module = types.SimpleNamespace(open=lambda path: FakeImage())

    def fake_import(name, *args, **kwargs):
        if name == "rembg":
            return types.SimpleNamespace(remove=lambda image, **kwargs: FakeImage())
        if name == "PIL":
            return types.SimpleNamespace(Image=fake_image_module)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = BgRemove().execute(
        {
            "input_path": str(input_path),
            "output_path": output_path,
        }
    )

    assert result.success
    assert result.data["output_path"] == output_path
    assert result.artifacts == [output_path]
    _assert_output_payload_matches_schema(
        BgRemove,
        result.data,
        {"input", "output", "output_path", "model", "alpha_matting", "bg_color"},
    )


@pytest.mark.parametrize("output_kind", ["missing", "relative", "absolute"])
def test_upscale_requires_project_output_path_before_upsampler(
    output_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "small.png"
    input_path.write_bytes(b"image")
    calls: list[tuple[object, ...]] = []

    def fake_upscale_image(self, *args: Any, **kwargs: Any) -> dict[str, int]:
        calls.append(args)
        return {"output_width": 16, "output_height": 16}

    monkeypatch.setattr(Upscale, "_upscale_image", fake_upscale_image)

    inputs: dict[str, object] = {"input_path": str(input_path)}
    if output_kind == "relative":
        inputs["output_path"] = "upscaled.png"
    elif output_kind == "absolute":
        inputs["output_path"] = str(tmp_path / "upscaled.png")

    result = Upscale().execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert calls == []


def test_upscale_rejects_unknown_model_before_upsampler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "small.png"
    input_path.write_bytes(b"image")
    output_path = tmp_path / "projects" / "demo" / "assets" / "images" / "large.png"

    def fake_upscale_image(self, *args: Any, **kwargs: Any):
        output_path.write_bytes(b"large")
        return {"output_width": 16, "output_height": 16}

    monkeypatch.setattr(Upscale, "_upscale_image", fake_upscale_image)

    result = Upscale().execute(
        {
            "input_path": str(input_path),
            "output_path": "projects/demo/assets/images/large.png",
            "model": "not-a-real-model",
        }
    )

    assert result.success is False
    assert "Unknown model" in (result.error or "")


def test_upscale_rejects_unknown_scale_before_upsampler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "small.png"
    input_path.write_bytes(b"image")
    output_path = tmp_path / "projects" / "demo" / "assets" / "images" / "large.png"

    def fake_upscale_image(self, *args: Any, **kwargs: Any):
        output_path.write_bytes(b"large")
        return {"output_width": 16, "output_height": 16}

    monkeypatch.setattr(Upscale, "_upscale_image", fake_upscale_image)

    result = Upscale().execute(
        {
            "input_path": str(input_path),
            "output_path": "projects/demo/assets/images/large.png",
            "scale": 3,
        }
    )

    assert result.success is False
    assert "Unknown scale" in (result.error or "")


def test_upscale_success_payload_includes_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "small.png"
    input_path.write_bytes(b"image")
    output_path = "projects/demo/assets/images/large.png"

    def fake_upscale_image(self, *args: Any, **kwargs: Any) -> dict[str, int]:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"large")
        return {"output_width": 16, "output_height": 16}

    monkeypatch.setattr(Upscale, "_upscale_image", fake_upscale_image)

    result = Upscale().execute(
        {
            "input_path": str(input_path),
            "output_path": output_path,
        }
    )

    assert result.success
    assert result.data["output_path"] == output_path
    assert result.artifacts == [output_path]
    _assert_output_payload_matches_schema(
        Upscale,
        result.data,
        {
            "input",
            "output",
            "output_path",
            "scale",
            "model",
            "face_enhance",
            "type",
            "output_width",
            "output_height",
            "total_frames",
            "fps",
        },
    )


@pytest.mark.parametrize("output_kind", ["missing", "relative", "absolute"])
def test_face_restore_requires_project_output_path_before_restorer_load(
    output_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "face.png"
    input_path.write_bytes(b"image")
    original_import = builtins.__import__
    restorer_inits: list[dict[str, object]] = []

    class FakeRestorer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            restorer_inits.append(kwargs)

        def enhance(self, *args: Any, **kwargs: Any):
            return None, [object()], object()

    fake_cv2 = types.SimpleNamespace(
        IMREAD_COLOR=1,
        imread=lambda *args, **kwargs: object(),
        imwrite=lambda path, img: (Path(path).write_bytes(b"restored"), True)[1],
    )

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            return fake_cv2
        if name == "gfpgan":
            return types.SimpleNamespace(GFPGANer=FakeRestorer)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    inputs: dict[str, object] = {"input_path": str(input_path)}
    if output_kind == "relative":
        inputs["output_path"] = "restored.png"
        forbidden_output = tmp_path / "restored.png"
    elif output_kind == "absolute":
        forbidden_output = tmp_path / "restored.png"
        inputs["output_path"] = str(forbidden_output)
    else:
        forbidden_output = tmp_path / "face_restored.png"

    result = FaceRestore().execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert restorer_inits == []
    assert not forbidden_output.exists()


def test_face_restore_rejects_unknown_model_before_restorer_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "face.png"
    input_path.write_bytes(b"image")
    original_import = builtins.__import__

    class FakeRestorer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def enhance(self, *args: Any, **kwargs: Any):
            return None, [object()], object()

    fake_cv2 = types.SimpleNamespace(
        IMREAD_COLOR=1,
        imread=lambda *args, **kwargs: object(),
        imwrite=lambda path, img: (Path(path).write_bytes(b"restored"), True)[1],
    )

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            return fake_cv2
        if name == "gfpgan":
            return types.SimpleNamespace(GFPGANer=FakeRestorer)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = FaceRestore().execute(
        {
            "input_path": str(input_path),
            "output_path": "projects/demo/assets/images/restored.png",
            "model": "not-a-real-model",
        }
    )

    assert result.success is False
    assert "Unknown model" in (result.error or "")


def test_face_restore_success_payload_includes_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "face.png"
    input_path.write_bytes(b"image")
    output_path = "projects/demo/assets/images/restored.png"
    original_import = builtins.__import__

    class FakeRestorer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def enhance(self, *args: Any, **kwargs: Any):
            return None, [object()], object()

    fake_cv2 = types.SimpleNamespace(
        IMREAD_COLOR=1,
        imread=lambda *args, **kwargs: object(),
        imwrite=lambda path, img: (Path(path).write_bytes(b"restored"), True)[1],
    )

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            return fake_cv2
        if name == "gfpgan":
            return types.SimpleNamespace(GFPGANer=FakeRestorer)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = FaceRestore().execute(
        {
            "input_path": str(input_path),
            "output_path": output_path,
        }
    )

    assert result.success
    assert result.data["output_path"] == output_path
    assert result.artifacts == [output_path]
    _assert_output_payload_matches_schema(
        FaceRestore,
        result.data,
        {
            "input",
            "output",
            "output_path",
            "model",
            "faces_detected",
            "upscale",
            "fidelity",
            "bg_upsampler",
        },
    )


def test_eye_enhance_rejects_unknown_operation_before_fallback_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "speaker.mp4"
    input_path.write_bytes(b"video")
    output_path = tmp_path / "projects" / "demo" / "renders" / "eyes.mp4"

    monkeypatch.setattr(EyeEnhance, "_has_mediapipe", lambda self: False)
    monkeypatch.setattr(EyeEnhance, "_has_opencv", lambda self: False)
    monkeypatch.setattr(EyeEnhance, "run_command", lambda *args, **kwargs: output_path.write_bytes(b"eyes"))

    result = EyeEnhance().execute(
        {
            "input_path": str(input_path),
            "output_path": "projects/demo/renders/eyes.mp4",
            "operations": ["dark_circles", "not-a-real-operation"],
        }
    )

    assert result.success is False
    assert "Unknown operation" in (result.error or "")


@pytest.mark.parametrize("output_kind", ["missing", "relative", "absolute"])
def test_eye_enhance_requires_project_output_path_before_fallback_render(
    output_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "speaker.mp4"
    input_path.write_bytes(b"video")
    commands: list[list[str]] = []

    monkeypatch.setattr(EyeEnhance, "_has_mediapipe", lambda self: False)
    monkeypatch.setattr(EyeEnhance, "_has_opencv", lambda self: False)

    def fake_run_command(self: EyeEnhance, cmd: list[str], **kwargs: object) -> None:
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"eyes")

    monkeypatch.setattr(EyeEnhance, "run_command", fake_run_command)
    inputs: dict[str, object] = {
        "input_path": str(input_path),
        "operations": ["dark_circles"],
    }
    if output_kind == "relative":
        inputs["output_path"] = "eyes.mp4"
        forbidden_output = tmp_path / "eyes.mp4"
    elif output_kind == "absolute":
        forbidden_output = tmp_path / "eyes.mp4"
        inputs["output_path"] = str(forbidden_output)
    else:
        forbidden_output = tmp_path / "speaker_eye_enhanced.mp4"

    result = EyeEnhance().execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert commands == []
    assert not forbidden_output.exists()


@pytest.mark.parametrize(
    ("method_name", "has_mediapipe", "has_opencv"),
    [
        ("_enhance_mediapipe", True, True),
        ("_enhance_opencv_only", False, True),
    ],
)
def test_eye_enhance_non_ffmpeg_success_payload_includes_output_path(
    method_name: str,
    has_mediapipe: bool,
    has_opencv: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "speaker.mp4"
    input_path.write_bytes(b"video")
    output_path = "projects/demo/renders/eyes.mp4"
    calls: list[str] = []

    monkeypatch.setattr(EyeEnhance, "_has_mediapipe", lambda self: has_mediapipe)
    monkeypatch.setattr(EyeEnhance, "_has_opencv", lambda self: has_opencv)

    def fake_enhance(
        self: EyeEnhance,
        input_arg: Path,
        output_arg: Path,
        inputs: dict[str, Any],
    ) -> ToolResult:
        calls.append(method_name)
        output_arg.parent.mkdir(parents=True, exist_ok=True)
        output_arg.write_bytes(b"eyes")
        return ToolResult(
            success=True,
            data={
                "input": str(input_arg),
                "output": str(output_arg),
                "method": method_name,
                "frames_processed": 1,
                "frames_enhanced": 1,
                "operations": ["dark_circles"],
            },
            artifacts=[str(output_arg)],
        )

    monkeypatch.setattr(EyeEnhance, method_name, fake_enhance)

    result = EyeEnhance().execute(
        {
            "input_path": str(input_path),
            "output_path": output_path,
        }
    )

    assert calls == [method_name]
    assert result.success
    assert result.data["output_path"] == output_path
    assert result.artifacts == [output_path]
    _assert_output_payload_matches_schema(
        EyeEnhance,
        result.data,
        {
            "input",
            "output",
            "output_path",
            "method",
            "frames_processed",
            "frames_enhanced",
            "operations",
            "note",
        },
    )


def test_eye_enhance_idempotency_key_includes_output_and_encoding_parameters():
    tool = EyeEnhance()
    base = {
        "input_path": "speaker.mp4",
        "operations": ["dark_circles"],
        "dark_circle_intensity": 0.4,
        "eye_brighten_intensity": 0.3,
        "sharpen_intensity": 0.2,
        "output_path": "eyes-a.mp4",
        "codec": "libx264",
        "crf": 18,
    }

    base_key = tool.idempotency_key(base)
    for variant in (
        {"output_path": "eyes-b.mp4"},
        {"codec": "libx265"},
        {"crf": 22},
    ):
        assert tool.idempotency_key({**base, **variant}) != base_key


def test_eye_enhance_ffmpeg_fallback_uses_requested_codec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "speaker.mp4"
    input_path.write_bytes(b"video")
    output_path = tmp_path / "projects" / "demo" / "renders" / "eyes.mp4"
    captured_cmd: list[str] = []

    monkeypatch.setattr(EyeEnhance, "_has_mediapipe", lambda self: False)
    monkeypatch.setattr(EyeEnhance, "_has_opencv", lambda self: False)

    def fake_run_command(self: EyeEnhance, cmd: list[str], **kwargs: Any):
        captured_cmd[:] = cmd
        output_path.write_bytes(b"eyes")

    monkeypatch.setattr(EyeEnhance, "run_command", fake_run_command)

    result = EyeEnhance().execute(
        {
            "input_path": str(input_path),
            "output_path": "projects/demo/renders/eyes.mp4",
            "codec": "libx265",
            "crf": 22,
        }
    )

    assert result.success
    assert captured_cmd[captured_cmd.index("-c:v") + 1] == "libx265"


@pytest.mark.parametrize(
    ("tool", "input_name", "output_path", "extra_inputs"),
    [
        (
            FaceEnhance(),
            "speaker.mp4",
            "projects/demo/renders/face-enhanced.mp4",
            {"preset": "sharpen"},
        ),
        (
            ColorGrade(),
            "scene.mp4",
            "projects/demo/renders/graded.mp4",
            {"profile": "neutral"},
        ),
        (
            AudioEnhance(),
            "voice.wav",
            "projects/demo/assets/audio/enhanced.wav",
            {"preset": "normalize_only"},
        ),
        (
            EyeEnhance(),
            "speaker.mp4",
            "projects/demo/renders/eyes.mp4",
            {},
        ),
    ],
)
def test_ffmpeg_enhancement_success_payload_includes_output_path(
    tool: Any,
    input_name: str,
    output_path: str,
    extra_inputs: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / input_name
    input_path.write_bytes(b"media")

    def fake_run_command(self: Any, cmd: list[str], **kwargs: object) -> None:
        Path(cmd[-1]).write_bytes(b"enhanced")

    monkeypatch.setattr(type(tool), "run_command", fake_run_command)
    if isinstance(tool, EyeEnhance):
        monkeypatch.setattr(EyeEnhance, "_has_mediapipe", lambda self: False)
        monkeypatch.setattr(EyeEnhance, "_has_opencv", lambda self: False)

    result = tool.execute(
        {
            "input_path": str(input_path),
            "output_path": output_path,
            **extra_inputs,
        }
    )

    assert result.success
    assert result.data["output_path"] == output_path
    assert result.artifacts == [output_path]
    if isinstance(tool, FaceEnhance):
        _assert_output_payload_matches_schema(
            FaceEnhance,
            result.data,
            {"input", "output", "output_path", "filter", "preset"},
        )
    elif isinstance(tool, ColorGrade):
        _assert_output_payload_matches_schema(
            ColorGrade,
            result.data,
            {"input", "output", "output_path", "filter", "profile", "lut", "intensity"},
        )
    elif isinstance(tool, EyeEnhance):
        _assert_output_payload_matches_schema(
            EyeEnhance,
            result.data,
            {
                "input",
                "output",
                "output_path",
                "method",
                "frames_processed",
                "frames_enhanced",
                "operations",
                "note",
            },
        )


@pytest.mark.parametrize("output_kind", ["missing", "relative", "absolute"])
def test_audio_enhance_requires_project_output_path_before_ffmpeg(
    output_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "voice.wav"
    input_path.write_bytes(b"audio")
    commands: list[list[str]] = []

    def fake_run_command(self: AudioEnhance, cmd: list[str], **kwargs: object) -> None:
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"enhanced")

    monkeypatch.setattr(AudioEnhance, "run_command", fake_run_command)
    inputs: dict[str, object] = {
        "input_path": str(input_path),
        "preset": "normalize_only",
    }
    if output_kind == "relative":
        inputs["output_path"] = "enhanced.wav"
        forbidden_output = tmp_path / "enhanced.wav"
    elif output_kind == "absolute":
        forbidden_output = tmp_path / "enhanced.wav"
        inputs["output_path"] = str(forbidden_output)
    else:
        forbidden_output = tmp_path / "voice_enhanced.wav"

    result = AudioEnhance().execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert commands == []
    assert not forbidden_output.exists()


@pytest.mark.parametrize(
    ("tool", "input_name", "extra_inputs"),
    [
        (FaceEnhance(), "speaker.mp4", {"preset": "sharpen"}),
        (ColorGrade(), "scene.mp4", {"profile": "neutral"}),
        (AudioEnhance(), "voice.wav", {"preset": "normalize_only"}),
        (EyeEnhance(), "speaker.mp4", {}),
    ],
)
def test_ffmpeg_enhancement_tools_fail_when_expected_output_is_missing(
    tool: Any,
    input_name: str,
    extra_inputs: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / input_name
    output_path = "projects/demo/renders/enhanced.mp4"
    input_path.write_bytes(b"media")

    def fake_run_command(*args: Any, **kwargs: Any):
        return object()

    monkeypatch.setattr(type(tool), "run_command", fake_run_command)
    if isinstance(tool, EyeEnhance):
        monkeypatch.setattr(EyeEnhance, "_has_mediapipe", lambda self: False)
        monkeypatch.setattr(EyeEnhance, "_has_opencv", lambda self: False)

    result = tool.execute(
        {
            "input_path": str(input_path),
            "output_path": output_path,
            **extra_inputs,
        }
    )

    assert result.success is False
    assert "output" in (result.error or "").lower()
    assert output_path in (result.error or "")


def test_face_restore_fails_when_cv2_write_does_not_create_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "face.png"
    output_path = tmp_path / "projects" / "demo" / "assets" / "images" / "restored.png"
    input_path.write_bytes(b"image")
    original_import = builtins.__import__

    class FakeRestorer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def enhance(self, *args: Any, **kwargs: Any):
            return None, [object()], object()

    fake_cv2 = types.SimpleNamespace(
        IMREAD_COLOR=1,
        imread=lambda *args, **kwargs: object(),
        imwrite=lambda *args, **kwargs: False,
    )

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            return fake_cv2
        if name == "gfpgan":
            return types.SimpleNamespace(GFPGANer=FakeRestorer)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = FaceRestore().execute(
        {
            "input_path": str(input_path),
            "output_path": "projects/demo/assets/images/restored.png",
        }
    )

    assert result.success is False
    assert "projects/demo/assets/images/restored.png" in (result.error or "")


def test_upscale_fails_when_cv2_image_write_does_not_create_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "small.png"
    output_path = tmp_path / "projects" / "demo" / "assets" / "images" / "upscaled.png"
    input_path.write_bytes(b"image")
    original_import = builtins.__import__

    class FakeImage:
        shape = (12, 16, 3)

    class FakeUpsampler:
        def enhance(self, *args: Any, **kwargs: Any):
            return FakeImage(), None

    fake_cv2 = types.SimpleNamespace(
        IMREAD_UNCHANGED=-1,
        imread=lambda *args, **kwargs: object(),
        imwrite=lambda *args, **kwargs: False,
    )

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            return fake_cv2
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(
        Upscale,
        "_build_upsampler",
        lambda self, *args, **kwargs: FakeUpsampler(),
    )

    result = Upscale().execute(
        {"input_path": str(input_path), "output_path": "projects/demo/assets/images/upscaled.png"}
    )

    assert result.success is False
    assert "projects/demo/assets/images/upscaled.png" in (result.error or "")


def test_upscale_video_fails_when_ffmpeg_extracts_no_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "clip.mp4"
    input_path.write_bytes(b"video")
    original_import = builtins.__import__

    fake_cv2 = types.SimpleNamespace()

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            return fake_cv2
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(Upscale, "_get_video_fps", lambda self, path: 24.0)
    monkeypatch.setattr(Upscale, "_build_upsampler", lambda self, *args, **kwargs: object())
    monkeypatch.setattr(Upscale, "run_command", lambda *args, **kwargs: object())

    result = Upscale().execute(
        {"input_path": str(input_path), "output_path": "projects/demo/renders/upscaled.mp4"}
    )

    assert result.success is False
    assert "frame" in (result.error or "").lower()
