from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import jsonschema

from tools.audio.audio_mixer import AudioMixer
from tools.base_tool import ToolResult, ToolStatus
from tools.capture.cap_recorder import CapRecorder
from tools.capture.screen_recorder import ScreenRecorder
from tools.capture.screen_capture_selector import ScreenCaptureSelector
from tools.character.character_animation import (
    ActionTimelineCompiler,
    CharacterAnimationReviewer,
    CharacterRigRenderer,
    CharacterSpecGenerator,
    PoseLibraryBuilder,
    SvgRigBuilder,
)
from tools.compliance.compliance_check import ComplianceCheck
from tools.enhancement.eye_enhance import EyeEnhance
from tools.graphics.code_snippet import CodeSnippet
from tools.graphics.diagram_gen import DiagramGen
from tools.graphics.flux_image import FluxImage
from tools.graphics.google_imagen import GoogleImagen
from tools.graphics.math_animate import MathAnimate
from tools.graphics.openai_image import OpenAIImage
from tools.graphics.recraft_image import RecraftImage


def _project_graphics_output(name: str) -> str:
    return f"projects/test-graphics/assets/images/{name}"


def test_screen_recorder_reports_ffmpeg_failure_even_if_partial_file_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    output_path = Path("projects/test-screen-demo/renders/recording.mp4")

    def fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "ffmpeg":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"partial mp4")
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr="x11grab input failed",
            )
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, stdout="1280,720\n", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("tools.capture.screen_recorder.platform.system", lambda: "Linux")
    monkeypatch.setattr("tools.capture.screen_recorder.subprocess.run", fake_run)
    monkeypatch.setenv("DISPLAY", ":0.0")

    result = ScreenRecorder().execute(
        {
            "output_path": "projects/test-screen-demo/renders/recording.mp4",
            "duration_seconds": 1,
            "capture_audio": False,
        }
    )

    assert result.success is False
    assert "ffmpeg" in (result.error or "").lower()
    assert "x11grab input failed" in (result.error or "")


@pytest.mark.parametrize("output_path", ["recording.mp4", "/tmp/recording.mp4"])
def test_screen_recorder_requires_project_output_path_before_ffmpeg(
    output_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

    monkeypatch.setattr("tools.capture.screen_recorder.platform.system", lambda: "Linux")
    monkeypatch.setattr("tools.capture.screen_recorder.subprocess.run", fake_run)

    result = ScreenRecorder().execute(
        {
            "output_path": output_path,
            "duration_seconds": 1,
            "capture_audio": False,
        }
    )

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert calls == []


@pytest.mark.parametrize(
    "variant",
    [
        {"output_path": "recording-a.mp4"},
        {"duration_seconds": 90},
        {"fps": 60},
        {"capture_audio": False},
        {"region": {"x": 10, "y": 20, "width": 640, "height": 360}},
        {"screen_index": 1},
    ],
)
def test_screen_recorder_idempotency_key_includes_capture_parameters(
    variant: dict[str, object],
) -> None:
    tool = ScreenRecorder()
    base = {
        "output_path": "recording.mp4",
        "duration_seconds": 30,
        "fps": 30,
        "capture_audio": True,
        "region": {"x": 0, "y": 0, "width": 1280, "height": 720},
        "screen_index": 0,
    }

    assert tool.idempotency_key(base) != tool.idempotency_key({**base, **variant})


@pytest.mark.parametrize("output_dir", [None, "recording.mp4", "/tmp/recording.mp4"])
def test_cap_recorder_pick_latest_requires_project_output_dir_before_lookup(
    output_dir: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lookup_calls: list[str] = []
    recording_calls: list[object] = []

    def fake_recordings_dir() -> Path:
        lookup_calls.append("recordings_dir")
        return tmp_path

    def fake_recent_recordings(*args: object, **kwargs: object) -> list[dict[str, object]]:
        recording_calls.append((args, kwargs))
        return []

    monkeypatch.setattr("tools.capture.cap_recorder._find_cap_recordings_dir", fake_recordings_dir)
    monkeypatch.setattr("tools.capture.cap_recorder._get_recent_recordings", fake_recent_recordings)

    inputs: dict[str, object] = {"operation": "pick_latest"}
    if output_dir is not None:
        inputs["output_dir"] = output_dir

    result = CapRecorder().execute(inputs)

    assert result.success is False
    assert "output_dir" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert lookup_calls == []
    assert recording_calls == []


def test_cap_recorder_pick_latest_copies_to_project_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    recordings_dir = tmp_path / "cap-recordings"
    recordings_dir.mkdir()
    source = recordings_dir / "recording.mp4"
    source.write_bytes(b"cap-video")

    monkeypatch.setattr("tools.capture.cap_recorder._find_cap_recordings_dir", lambda: recordings_dir)
    monkeypatch.setattr(
        "tools.capture.cap_recorder._get_recent_recordings",
        lambda *args, **kwargs: [
            {
                "path": str(source),
                "name": "recording",
                "size_mb": 1.0,
                "modified": 1.0,
            }
        ],
    )

    result = CapRecorder().execute(
        {
            "operation": "pick_latest",
            "output_dir": "projects/test-screen-demo/renders/cap",
        }
    )

    copied = tmp_path / "projects/test-screen-demo/renders/cap/recording.mp4"
    assert result.success is True
    assert copied.read_bytes() == b"cap-video"
    assert result.data["output_dir"] == "projects/test-screen-demo/renders/cap"
    assert result.data["output_path"] == (
        "projects/test-screen-demo/renders/cap/recording.mp4"
    )
    assert result.artifacts == [
        "projects/test-screen-demo/renders/cap/recording.mp4"
    ]


def test_cap_recorder_pick_latest_honors_since_minutes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    recordings_dir = tmp_path / "cap-recordings"
    recordings_dir.mkdir()
    source = recordings_dir / "recording.mp4"
    source.write_bytes(b"cap-video")
    since_values: list[int] = []

    monkeypatch.setattr("tools.capture.cap_recorder._find_cap_recordings_dir", lambda: recordings_dir)

    def fake_recent_recordings(_recordings_dir: Path, *, since_seconds: int) -> list[dict[str, object]]:
        since_values.append(since_seconds)
        return [
            {
                "path": str(source),
                "name": "recording",
                "size_mb": 1.0,
                "modified": 1.0,
            }
        ]

    monkeypatch.setattr("tools.capture.cap_recorder._get_recent_recordings", fake_recent_recordings)

    result = CapRecorder().execute(
        {
            "operation": "pick_latest",
            "output_dir": "projects/test-screen-demo/renders/cap/latest.mp4",
            "since_minutes": 7,
        }
    )

    assert result.success is True
    assert since_values == [420]


def test_cap_recorder_schema_requires_output_dir_for_pick_latest() -> None:
    schema = CapRecorder.input_schema

    jsonschema.validate({"operation": "find_recordings"}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"operation": "pick_latest"}, schema)


@pytest.mark.parametrize(
    ("base", "variant"),
    [
        (
            {
                "operation": "duck",
                "primary_audio": "speech-a.wav",
                "secondary_audio": "music.wav",
                "duck_level": -12,
            },
            {"output_path": "ducked-a.wav"},
        ),
        (
            {
                "operation": "duck",
                "primary_audio": "speech-a.wav",
                "secondary_audio": "music.wav",
                "duck_level": -12,
            },
            {"primary_audio": "speech-b.wav"},
        ),
        (
            {
                "operation": "full_mix",
                "tracks": [
                    {"path": "speech.wav", "role": "speech"},
                    {"path": "music.wav", "role": "music"},
                ],
                "ducking": {"enabled": True},
                "normalize": True,
                "target_lufs": -16,
                "target_total_duration_seconds": 30,
            },
            {"target_total_duration_seconds": 15},
        ),
        (
            {
                "operation": "duck",
                "primary_audio": "speech.wav",
                "secondary_audio": "music.wav",
                "music_volume_schedule": [{"t_seconds": 0, "gain_db": -12}],
            },
            {"music_volume_schedule": [{"t_seconds": 0, "gain_db": -3}]},
        ),
        (
            {
                "operation": "segmented_music",
                "video_path": "assembled.mp4",
                "music_path": "music-a.mp3",
                "segments": [{"start": 0, "end": 5}],
                "music_volume": 0.2,
                "fade_duration": 0.5,
            },
            {"segments": [{"start": 5, "end": 10}]},
        ),
    ],
)
def test_audio_mixer_idempotency_key_includes_all_output_shaping_inputs(
    base: dict[str, object],
    variant: dict[str, object],
) -> None:
    tool = AudioMixer()

    assert tool.idempotency_key(base) != tool.idempotency_key({**base, **variant})


@pytest.mark.parametrize("output_kind", ["missing", "relative", "absolute"])
def test_audio_mixer_mix_requires_project_output_path_before_ffmpeg(
    output_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    speech_path = tmp_path / "speech.wav"
    music_path = tmp_path / "music.wav"
    speech_path.write_bytes(b"speech")
    music_path.write_bytes(b"music")
    commands: list[list[str]] = []

    def fake_run_command(self: AudioMixer, cmd: list[str], **kwargs: object) -> None:
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"mixed")

    monkeypatch.setattr(AudioMixer, "run_command", fake_run_command)
    inputs: dict[str, object] = {
        "operation": "mix",
        "tracks": [
            {"path": str(speech_path), "role": "speech"},
            {"path": str(music_path), "role": "music", "volume": 0.3},
        ],
    }
    if output_kind == "relative":
        inputs["output_path"] = "mixed.wav"
        forbidden_output = tmp_path / "mixed.wav"
    elif output_kind == "absolute":
        forbidden_output = tmp_path / "mixed.wav"
        inputs["output_path"] = str(forbidden_output)
    else:
        forbidden_output = tmp_path / "mixed_audio.wav"

    result = AudioMixer().execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert commands == []
    assert not forbidden_output.exists()


@pytest.mark.parametrize("output_kind", ["missing", "relative", "absolute"])
def test_audio_mixer_duck_requires_project_output_path_before_ffmpeg(
    output_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    speech_path = tmp_path / "speech.wav"
    music_path = tmp_path / "music.wav"
    speech_path.write_bytes(b"speech")
    music_path.write_bytes(b"music")
    commands: list[list[str]] = []

    def fake_run_command(self: AudioMixer, cmd: list[str], **kwargs: object) -> None:
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"ducked")

    monkeypatch.setattr(AudioMixer, "run_command", fake_run_command)
    inputs: dict[str, object] = {
        "operation": "duck",
        "primary_audio": str(speech_path),
        "secondary_audio": str(music_path),
    }
    if output_kind == "relative":
        inputs["output_path"] = "ducked.wav"
        forbidden_output = tmp_path / "ducked.wav"
    elif output_kind == "absolute":
        forbidden_output = tmp_path / "ducked.wav"
        inputs["output_path"] = str(forbidden_output)
    else:
        forbidden_output = tmp_path / "ducked_audio.wav"

    result = AudioMixer().execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert commands == []
    assert not forbidden_output.exists()


@pytest.mark.parametrize("output_kind", ["missing", "relative", "absolute"])
def test_audio_mixer_extract_requires_project_output_path_before_ffmpeg(
    output_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "source.mp4"
    input_path.write_bytes(b"video")
    commands: list[list[str]] = []

    def fake_run_command(self: AudioMixer, cmd: list[str], **kwargs: object) -> None:
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"audio")

    monkeypatch.setattr(AudioMixer, "run_command", fake_run_command)
    inputs: dict[str, object] = {
        "operation": "extract",
        "input_path": str(input_path),
    }
    if output_kind == "relative":
        inputs["output_path"] = "extracted.wav"
        forbidden_output = tmp_path / "extracted.wav"
    elif output_kind == "absolute":
        forbidden_output = tmp_path / "extracted.wav"
        inputs["output_path"] = str(forbidden_output)
    else:
        forbidden_output = input_path.with_suffix(".wav")

    result = AudioMixer().execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert commands == []
    assert not forbidden_output.exists()


@pytest.mark.parametrize("output_kind", ["missing", "relative", "absolute"])
def test_audio_mixer_segmented_music_requires_project_output_path_before_ffmpeg(
    output_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "assembled.mp4"
    music_path = tmp_path / "music.mp3"
    video_path.write_bytes(b"video")
    music_path.write_bytes(b"music")
    commands: list[list[str]] = []

    def fake_run_command(self: AudioMixer, cmd: list[str], **kwargs: object) -> object:
        commands.append(cmd)
        if cmd[0] == "ffprobe":
            return SimpleNamespace(stdout="10\n")
        Path(cmd[-1]).write_bytes(b"video-with-music")
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(AudioMixer, "run_command", fake_run_command)
    inputs: dict[str, object] = {
        "operation": "segmented_music",
        "video_path": str(video_path),
        "music_path": str(music_path),
        "segments": [{"start": 0, "end": 5}],
    }
    if output_kind == "relative":
        inputs["output_path"] = "final-with-music.mp4"
        forbidden_output = tmp_path / "final-with-music.mp4"
    elif output_kind == "absolute":
        forbidden_output = tmp_path / "final-with-music.mp4"
        inputs["output_path"] = str(forbidden_output)
    else:
        forbidden_output = tmp_path / "segmented_music_output.mp4"

    result = AudioMixer().execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert commands == []
    assert not forbidden_output.exists()


@pytest.mark.parametrize("output_kind", ["missing", "relative", "absolute"])
def test_audio_mixer_full_mix_requires_project_output_path_before_ffmpeg(
    output_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    speech_path = tmp_path / "speech.wav"
    music_path = tmp_path / "music.wav"
    speech_path.write_bytes(b"speech")
    music_path.write_bytes(b"music")
    commands: list[list[str]] = []

    def fake_run_command(self: AudioMixer, cmd: list[str], **kwargs: object) -> None:
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"mixed")

    monkeypatch.setattr(AudioMixer, "run_command", fake_run_command)
    monkeypatch.setattr(AudioMixer, "_probe_audio_duration", lambda self, path: 10.0)
    inputs: dict[str, object] = {
        "operation": "full_mix",
        "tracks": [
            {"path": str(speech_path), "role": "speech"},
            {"path": str(music_path), "role": "music"},
        ],
    }
    if output_kind == "relative":
        inputs["output_path"] = "full-mix.wav"
        forbidden_output = tmp_path / "full-mix.wav"
    elif output_kind == "absolute":
        forbidden_output = tmp_path / "full-mix.wav"
        inputs["output_path"] = str(forbidden_output)
    else:
        forbidden_output = tmp_path / "full_mix_output.wav"

    result = AudioMixer().execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert commands == []
    assert not forbidden_output.exists()


@pytest.mark.parametrize(
    ("inputs", "output_path"),
    [
        (
            {
                "operation": "mix",
                "tracks": [
                    {"path": "speech.wav", "role": "speech"},
                    {"path": "music.wav", "role": "music", "volume": 0.3},
                ],
            },
            "projects/demo/assets/audio/mixed.wav",
        ),
        (
            {
                "operation": "duck",
                "primary_audio": "speech.wav",
                "secondary_audio": "music.wav",
            },
            "projects/demo/assets/audio/ducked.wav",
        ),
        (
            {
                "operation": "extract",
                "input_path": "source.mp4",
            },
            "projects/demo/assets/audio/extracted.wav",
        ),
        (
            {
                "operation": "full_mix",
                "tracks": [
                    {"path": "speech.wav", "role": "speech"},
                    {"path": "music.wav", "role": "music"},
                ],
            },
            "projects/demo/assets/audio/full-mix.wav",
        ),
        (
            {
                "operation": "segmented_music",
                "video_path": "assembled.mp4",
                "music_path": "music.wav",
                "segments": [{"start": 0, "end": 5}],
            },
            "projects/demo/renders/final-with-music.mp4",
        ),
    ],
)
def test_audio_mixer_success_payload_includes_output_path(
    inputs: dict[str, object],
    output_path: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("speech.wav", "music.wav", "source.mp4", "assembled.mp4"):
        (tmp_path / name).write_bytes(b"media")

    def fake_run_command(self: AudioMixer, cmd: list[str], **kwargs: object) -> SimpleNamespace:
        if cmd[0] == "ffprobe":
            return SimpleNamespace(stdout="10\n", stderr="")
        Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[-1]).write_bytes(b"mixed")
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(AudioMixer, "run_command", fake_run_command)
    monkeypatch.setattr(AudioMixer, "_probe_audio_duration", lambda self, path: 10.0)

    result = AudioMixer().execute({**inputs, "output_path": output_path})

    assert result.success
    assert result.data["output_path"] == output_path
    assert result.artifacts == [output_path]


def test_default_character_timeline_uses_default_design_character_id() -> None:
    character_design = CharacterSpecGenerator().execute({}).data["character_design"]
    default_character_id = character_design["characters"][0]["id"]
    scene_plan = {
        "version": "1.0",
        "scenes": [
            {
                "id": "scene-1",
                "start_seconds": 0,
                "end_seconds": 2,
                "description": "Default character reacts to a small discovery.",
            }
        ],
    }

    timeline = ActionTimelineCompiler().execute({"scene_plan": scene_plan}).data["action_timeline"]
    action_character_ids = {
        action["character_id"]
        for scene in timeline["scenes"]
        for action in scene["actions"]
    }

    assert action_character_ids == {default_character_id}


def test_character_spec_generator_rejects_non_finite_artifact_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    output_path = Path("projects/demo/artifacts/character_design.json")

    result = CharacterSpecGenerator().execute(
        {
            "characters": [{"id": "mouse", "required_actions": ["idle"]}],
            "brief": math.nan,
            "output_path": str(output_path),
        }
    )

    assert result.success is False
    assert "strict JSON" in (result.error or "")
    assert not (tmp_path / output_path).exists()


def test_character_spec_generator_success_payload_includes_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    output_path = "projects/demo/artifacts/character_design.json"

    result = CharacterSpecGenerator().execute(
        {
            "characters": [{"id": "mouse", "required_actions": ["idle"]}],
            "output_path": output_path,
        }
    )

    assert result.success is True
    assert result.data["output_path"] == output_path
    assert result.artifacts == [output_path]
    assert (tmp_path / output_path).exists()


@pytest.mark.parametrize(
    ("tool", "inputs", "artifact_key", "output_path"),
    [
        (
            SvgRigBuilder(),
            {
                "character_design": {
                    "version": "1.0",
                    "characters": [{"id": "mouse", "required_actions": ["idle"]}],
                }
            },
            "rig_plan",
            "projects/demo/artifacts/rig_plan.json",
        ),
        (
            PoseLibraryBuilder(),
            {
                "rig_plan": {
                    "version": "1.0",
                    "characters": [{"character_id": "mouse", "required_actions": ["idle"]}],
                }
            },
            "pose_library",
            "projects/demo/artifacts/pose_library.json",
        ),
        (
            ActionTimelineCompiler(),
            {
                "scene_plan": {
                    "version": "1.0",
                    "scenes": [
                        {
                            "id": "scene-1",
                            "start_seconds": 0,
                            "end_seconds": 2,
                            "description": "Mouse notices a seed.",
                        }
                    ],
                },
                "character_ids": ["mouse"],
            },
            "action_timeline",
            "projects/demo/artifacts/action_timeline.json",
        ),
        (
            CharacterAnimationReviewer(),
            {
                "rig_plan": {
                    "version": "1.0",
                    "characters": [{"character_id": "mouse", "joints": {"head": {}}}],
                },
                "pose_library": {
                    "version": "1.0",
                    "characters": [{"character_id": "mouse", "poses": {"idle": {}}}],
                },
                "action_timeline": {
                    "version": "1.0",
                    "fps": 30,
                    "scenes": [{"scene_id": "scene-1", "actions": [{"pose": "idle"}]}],
                },
                "preview_path": "projects/demo/renders/preview.html",
            },
            "character_qa_report",
            "projects/demo/artifacts/character_qa_report.json",
        ),
    ],
)
def test_character_json_success_payload_matches_output_schema(
    tool: Any,
    inputs: dict[str, object],
    artifact_key: str,
    output_path: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    preview_path = tmp_path / "projects/demo/renders/preview.html"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_text("<html></html>", encoding="utf-8")

    output_properties = tool.output_schema["properties"]
    assert {artifact_key, "output_path"} <= set(output_properties)

    result = tool.execute({**inputs, "output_path": output_path})

    assert result.success is True
    assert result.data["output_path"] == output_path
    assert result.artifacts == [output_path]
    assert (tmp_path / output_path).exists()
    jsonschema.validate(instance=result.data, schema=tool.output_schema)


@pytest.mark.parametrize("output_kind", ["missing", "relative", "absolute"])
def test_character_rig_renderer_requires_project_output_path_before_writing_preview(
    output_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    inputs: dict[str, object] = {
        "action_timeline": {
            "scenes": [
                {
                    "id": "scene-1",
                    "end_seconds": 2,
                    "actions": [{"character_id": "mouse"}],
                }
            ]
        },
        "rig_plan": {"characters": [{"character_id": "mouse"}]},
        "pose_library": {"characters": [{"character_id": "mouse"}]},
    }
    if output_kind == "relative":
        inputs["output_path"] = "preview.html"
        forbidden_preview = tmp_path / "preview.html"
    elif output_kind == "absolute":
        forbidden_preview = tmp_path / "preview.html"
        inputs["output_path"] = str(forbidden_preview)
    else:
        forbidden_preview = tmp_path / "projects" / "character-preview" / "preview.html"

    result = CharacterRigRenderer().execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert not forbidden_preview.exists()
    assert not (forbidden_preview.parent / "hyperframes" / "hyperframes.json").exists()


def test_character_rig_renderer_schema_requires_output_path() -> None:
    schema = CharacterRigRenderer.input_schema

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"action_timeline": {"scenes": []}}, schema)


def test_character_rig_renderer_success_payload_includes_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    output_path = "projects/demo/renders/preview.html"

    result = CharacterRigRenderer().execute(
        {
            "action_timeline": {
                "scenes": [
                    {
                        "id": "scene-1",
                        "end_seconds": 2,
                        "actions": [{"character_id": "mouse"}],
                    }
                ]
            },
            "rig_plan": {"characters": [{"character_id": "mouse"}]},
            "pose_library": {"characters": [{"character_id": "mouse"}]},
            "output_path": output_path,
        }
    )

    assert result.success is True
    assert result.data["output_path"] == output_path
    assert result.data["preview_path"] == output_path
    assert result.artifacts[0] == output_path
    assert (tmp_path / output_path).exists()


def test_character_rig_renderer_success_payload_includes_explicit_secondary_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    output_path = "projects/demo/renders/preview.html"
    workspace_path = "projects/demo/renders/custom-hyperframes"
    video_output_path = "projects/demo/renders/custom-preview.mp4"

    def fake_render_preview_mp4(
        preview_path: Path,
        video_path: Path,
        duration_seconds: float,
        fps: int,
    ) -> None:
        video_path.write_bytes(b"mp4")

    monkeypatch.setattr(
        "tools.character.character_animation._render_preview_mp4",
        fake_render_preview_mp4,
    )

    result = CharacterRigRenderer().execute(
        {
            "action_timeline": {
                "scenes": [
                    {
                        "id": "scene-1",
                        "end_seconds": 2,
                        "actions": [{"character_id": "mouse"}],
                    }
                ]
            },
            "rig_plan": {"characters": [{"character_id": "mouse"}]},
            "pose_library": {"characters": [{"character_id": "mouse"}]},
            "output_path": output_path,
            "workspace_path": workspace_path,
            "video_output_path": video_output_path,
        }
    )

    assert result.success is True
    assert result.data["workspace_path"] == workspace_path
    assert result.data["hyperframes_workspace"] == workspace_path
    assert result.data["video_output_path"] == video_output_path
    assert result.data["video_path"] == video_output_path
    assert (tmp_path / workspace_path / "hyperframes.json").exists()
    assert (tmp_path / video_output_path).exists()


@pytest.mark.parametrize(
    ("field_name", "forbidden_name", "extra_inputs"),
    [
        ("workspace_path", "external-workspace", {}),
        ("video_output_path", "external-preview.mp4", {"render_video": True}),
    ],
)
def test_character_rig_renderer_rejects_external_secondary_outputs_before_writing(
    field_name: str,
    forbidden_name: str,
    extra_inputs: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    forbidden_path = tmp_path / forbidden_name
    render_calls: list[object] = []
    monkeypatch.setattr(
        "tools.character.character_animation._render_preview_mp4",
        lambda *args, **kwargs: render_calls.append((args, kwargs)),
    )

    result = CharacterRigRenderer().execute(
        {
            "action_timeline": {
                "scenes": [
                    {
                        "id": "scene-1",
                        "end_seconds": 2,
                        "actions": [{"character_id": "mouse"}],
                    }
                ]
            },
            "rig_plan": {"characters": [{"character_id": "mouse"}]},
            "pose_library": {"characters": [{"character_id": "mouse"}]},
            "output_path": "projects/demo/renders/preview.html",
            field_name: str(forbidden_path),
            **extra_inputs,
        }
    )

    assert result.success is False
    assert field_name in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert not (tmp_path / "projects/demo/renders/preview.html").exists()
    assert not forbidden_path.exists()
    assert render_calls == []


def test_character_rig_renderer_rejects_file_shaped_workspace_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    preview_path = Path("projects/demo/renders/preview.html")
    workspace_path = Path("projects/demo/renders/hyperframes.json")
    render_calls: list[object] = []
    monkeypatch.setattr(
        "tools.character.character_animation._render_preview_mp4",
        lambda *args, **kwargs: render_calls.append((args, kwargs)),
    )

    result = CharacterRigRenderer().execute(
        {
            "action_timeline": {
                "scenes": [
                    {
                        "id": "scene-1",
                        "end_seconds": 2,
                        "actions": [{"character_id": "mouse"}],
                    }
                ]
            },
            "rig_plan": {"characters": [{"character_id": "mouse"}]},
            "pose_library": {"characters": [{"character_id": "mouse"}]},
            "output_path": str(preview_path),
            "workspace_path": str(workspace_path),
        }
    )

    assert result.success is False
    assert "workspace_path" in (result.error or "")
    assert "must be a directory path" in (result.error or "")
    assert not (tmp_path / preview_path).exists()
    assert not (tmp_path / workspace_path).exists()
    assert render_calls == []


@pytest.mark.parametrize(
    ("tool", "base", "variant"),
    [
        (
            CharacterSpecGenerator(),
            {
                "characters": [{"id": "mouse", "required_actions": ["idle"]}],
                "brief": "A mouse learns to share.",
                "style": {"visual_style": "flat"},
            },
            {"output_path": "character-design-a.json"},
        ),
        (
            CharacterSpecGenerator(),
            {
                "characters": [{"id": "mouse", "required_actions": ["idle"]}],
                "brief": "A mouse learns to share.",
                "style": {"visual_style": "flat"},
            },
            {"brief": "A bird learns to share."},
        ),
        (
            SvgRigBuilder(),
            {
                "character_design": {
                    "characters": [{"id": "mouse", "required_actions": ["idle"]}]
                }
            },
            {"output_path": "rig-plan-a.json"},
        ),
        (
            SvgRigBuilder(),
            {
                "character_design": {
                    "characters": [{"id": "mouse", "required_actions": ["idle"]}]
                }
            },
            {
                "character_design": {
                    "characters": [{"id": "bird", "required_actions": ["wing_flap"]}]
                }
            },
        ),
        (
            PoseLibraryBuilder(),
            {
                "rig_plan": {
                    "characters": [
                        {"character_id": "mouse", "required_actions": ["idle"]}
                    ]
                }
            },
            {"output_path": "pose-library-a.json"},
        ),
        (
            PoseLibraryBuilder(),
            {
                "rig_plan": {
                    "characters": [
                        {"character_id": "mouse", "required_actions": ["idle"]}
                    ]
                }
            },
            {
                "rig_plan": {
                    "characters": [
                        {"character_id": "mouse", "required_actions": ["gesture"]}
                    ]
                }
            },
        ),
        (
            ActionTimelineCompiler(),
            {
                "scene_plan": {
                    "scenes": [
                        {
                            "id": "scene-1",
                            "start_seconds": 0,
                            "end_seconds": 2,
                            "description": "Mouse notices a seed.",
                        }
                    ]
                },
                "character_ids": ["mouse"],
                "fps": 24,
            },
            {"output_path": "action-timeline-a.json"},
        ),
        (
            ActionTimelineCompiler(),
            {
                "scene_plan": {
                    "scenes": [
                        {
                            "id": "scene-1",
                            "start_seconds": 0,
                            "end_seconds": 2,
                            "description": "Mouse notices a seed.",
                        }
                    ]
                },
                "character_ids": ["mouse"],
                "fps": 24,
            },
            {"character_ids": ["bird"]},
        ),
        (
            CharacterRigRenderer(),
            {
                "action_timeline": {
                    "scenes": [
                        {
                            "id": "scene-1",
                            "end_seconds": 2,
                            "actions": [{"character_id": "mouse"}],
                        }
                    ]
                },
                "rig_plan": {"characters": [{"character_id": "mouse"}]},
                "pose_library": {"characters": [{"character_id": "mouse"}]},
                "render_video": False,
                "duration_seconds": 2,
                "fps": 12,
            },
            {"output_path": "preview-a.html"},
        ),
        (
            CharacterRigRenderer(),
            {
                "action_timeline": {
                    "scenes": [
                        {
                            "id": "scene-1",
                            "end_seconds": 2,
                            "actions": [{"character_id": "mouse"}],
                        }
                    ]
                },
                "rig_plan": {"characters": [{"character_id": "mouse"}]},
                "pose_library": {"characters": [{"character_id": "mouse"}]},
                "render_video": False,
                "duration_seconds": 2,
                "fps": 12,
            },
            {"workspace_path": "character-hyperframes-a"},
        ),
        (
            CharacterRigRenderer(),
            {
                "action_timeline": {
                    "scenes": [
                        {
                            "id": "scene-1",
                            "end_seconds": 2,
                            "actions": [{"character_id": "mouse"}],
                        }
                    ]
                },
                "rig_plan": {"characters": [{"character_id": "mouse"}]},
                "pose_library": {"characters": [{"character_id": "mouse"}]},
                "render_video": False,
                "duration_seconds": 2,
                "fps": 12,
            },
            {"fps": 24},
        ),
        (
            CharacterAnimationReviewer(),
            {
                "rig_plan": {"characters": [{"joints": {"head": {}}}]},
                "pose_library": {"characters": [{"poses": {"idle": {}}}]},
                "action_timeline": {"scenes": [{"actions": [{"pose": "idle"}]}]},
                "preview_path": "preview-a.html",
                "review_level": "static",
                "browser_preview_checked": False,
                "frame_samples_checked": False,
            },
            {"output_path": "character-review-a.json"},
        ),
        (
            CharacterAnimationReviewer(),
            {
                "rig_plan": {"characters": [{"joints": {"head": {}}}]},
                "pose_library": {"characters": [{"poses": {"idle": {}}}]},
                "action_timeline": {"scenes": [{"actions": [{"pose": "idle"}]}]},
                "preview_path": "preview-a.html",
                "review_level": "static",
                "browser_preview_checked": False,
                "frame_samples_checked": False,
            },
            {"review_level": "final", "frame_samples_checked": True},
        ),
    ],
)
def test_character_animation_idempotency_keys_include_contract_inputs(
    tool,
    base: dict[str, object],
    variant: dict[str, object],
) -> None:
    assert tool.idempotency_key(base) != tool.idempotency_key({**base, **variant})


def test_compliance_structured_presence_invalid_min_count_returns_tool_error() -> None:
    checkpoint = {
        "id": "CP-PRESENCE",
        "check_type": "presence",
        "evaluation_method": "structural",
        "criterion": "structured criteria should drive this check",
        "failure_action": "revise",
        "structured": {
            "kind": "presence",
            "terms": ["brand mark"],
            "min_count": "two",
        },
    }

    try:
        result = ComplianceCheck().execute(
            {
                "stage_output": {"scenes": [{"description": "brand mark appears"}]},
                "checkpoint": checkpoint,
            }
        )
    except Exception as exc:  # pragma: no cover - assertion path documents bug shape
        pytest.fail(f"execute raised instead of returning ToolResult: {exc}")

    assert result.success is False
    assert "min_count" in (result.error or "")


def test_eye_enhance_status_degraded_when_only_ffmpeg_fallback_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(EyeEnhance, "_has_mediapipe", lambda self: False)
    monkeypatch.setattr(EyeEnhance, "_has_opencv", lambda self: False)
    monkeypatch.setattr(EyeEnhance, "check_dependencies", lambda self: None)

    assert EyeEnhance().get_status() == ToolStatus.DEGRADED


@pytest.mark.parametrize(
    ("tool", "base", "variant"),
    [
        (
            CodeSnippet(),
            {
                "code": "print('hello')",
                "language": "python",
                "theme": "monokai",
                "font_size": 20,
                "line_numbers": True,
                "padding": 40,
            },
            {"title": "example.py"},
        ),
        (
            CodeSnippet(),
            {
                "code": "print('hello')",
                "language": "python",
                "theme": "monokai",
                "font_size": 20,
                "line_numbers": True,
                "padding": 40,
            },
            {"output_path": "snippet-a.png"},
        ),
        (
            DiagramGen(),
            {
                "diagram_type": "boxes",
                "boxes": [{"label": "Input"}, {"label": "Output"}],
                "connections": [],
                "theme": "dark",
                "width": 1200,
                "height": 800,
            },
            {"output_path": "diagram-a.png"},
        ),
        (
            DiagramGen(),
            {
                "diagram_type": "boxes",
                "boxes": [{"label": "Input"}, {"label": "Output"}],
                "connections": [],
                "theme": "dark",
                "width": 1200,
                "height": 800,
            },
            {"connections": [{"from": 0, "to": 1, "label": "flow"}]},
        ),
        (
            MathAnimate(),
            {
                "scene_code": "class Demo(Scene):\n    def construct(self):\n        pass",
                "scene_name": "Demo",
                "quality": "low",
                "format": "mp4",
            },
            {"format": "gif"},
        ),
        (
            MathAnimate(),
            {
                "scene_code": "class Demo(Scene):\n    def construct(self):\n        pass",
                "scene_name": "Demo",
                "quality": "low",
                "format": "mp4",
            },
            {"output_path": "math-a.mp4"},
        ),
    ],
)
def test_graphics_idempotency_keys_include_output_shaping_inputs(
    tool: Any,
    base: dict[str, object],
    variant: dict[str, object],
) -> None:
    assert tool.idempotency_key(base) != tool.idempotency_key({**base, **variant})


def test_code_snippet_honors_forced_width_and_keys_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    tool = CodeSnippet()
    base = {
        "code": "print('hello')",
        "language": "python",
        "theme": "monokai",
        "font_size": 20,
        "line_numbers": True,
        "padding": 40,
        "width": 320,
    }

    result = tool.execute({**base, "output_path": _project_graphics_output("snippet.png")})

    assert result.success, result.error
    assert result.data["width"] == 320
    assert (tmp_path / result.data["output"]).exists()
    assert tool.idempotency_key(base) != tool.idempotency_key({**base, "width": 640})


@pytest.mark.parametrize(
    ("tool", "inputs", "output_path", "expected_properties"),
    [
        (
            CodeSnippet(),
            {"code": "print('hello')", "language": "python"},
            _project_graphics_output("snippet.png"),
            {"output", "output_path", "language", "theme", "width", "height", "line_count"},
        ),
        (
            DiagramGen(),
            {
                "diagram_type": "boxes",
                "boxes": [{"label": "Input"}, {"label": "Output"}],
                "connections": [{"from": 0, "to": 1, "label": "flow"}],
            },
            _project_graphics_output("diagram.png"),
            {"method", "output", "output_path", "box_count", "connection_count"},
        ),
    ],
)
def test_local_graphics_success_payload_matches_output_schema(
    tool: CodeSnippet | DiagramGen,
    inputs: dict[str, object],
    output_path: str,
    expected_properties: set[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    output_properties = tool.output_schema["properties"]
    assert expected_properties <= set(output_properties)

    result = tool.execute({**inputs, "output_path": output_path})

    assert result.success
    assert result.data["output_path"] == output_path
    assert result.artifacts == [output_path]
    jsonschema.validate(instance=result.data, schema=tool.output_schema)


@pytest.mark.parametrize("output_kind", ["missing", "relative", "absolute"])
def test_code_snippet_requires_project_output_path_before_rendering(
    output_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    inputs: dict[str, object] = {"code": "print('hello')", "language": "python"}
    if output_kind == "relative":
        inputs["output_path"] = "snippet.png"
        forbidden_output = tmp_path / "snippet.png"
    elif output_kind == "absolute":
        forbidden_output = tmp_path / "snippet.png"
        inputs["output_path"] = str(forbidden_output)
    else:
        forbidden_output = tmp_path / "code_snippet.png"

    result = CodeSnippet().execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert not forbidden_output.exists()


@pytest.mark.parametrize("output_kind", ["missing", "relative", "absolute"])
def test_diagram_gen_requires_project_output_path_before_mermaid_render(
    output_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    run_calls: list[list[str]] = []

    def fake_run_command(self: DiagramGen, cmd: list[str], *args: object, **kwargs: object) -> None:
        run_calls.append(cmd)
        output = Path(cmd[cmd.index("-o") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"png")

    monkeypatch.setattr(DiagramGen, "_has_mermaid", lambda self: True)
    monkeypatch.setattr(DiagramGen, "run_command", fake_run_command)

    inputs: dict[str, object] = {
        "diagram_type": "mermaid",
        "definition": "graph TD\n  A[Start] --> B[End]",
    }
    if output_kind == "relative":
        inputs["output_path"] = "diagram.png"
        forbidden_output = tmp_path / "diagram.png"
    elif output_kind == "absolute":
        forbidden_output = tmp_path / "diagram.png"
        inputs["output_path"] = str(forbidden_output)
    else:
        forbidden_output = tmp_path / "diagram.png"

    result = DiagramGen().execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert run_calls == []
    assert not forbidden_output.exists()


@pytest.mark.parametrize(
    ("tool_cls", "instance"),
    [
        (CodeSnippet, {"code": "print('hello')"}),
        (DiagramGen, {"diagram_type": "boxes", "boxes": []}),
    ],
)
def test_local_graphics_schemas_require_output_path(
    tool_cls: type[CodeSnippet] | type[DiagramGen],
    instance: dict[str, object],
) -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=instance, schema=tool_cls.input_schema)


@pytest.mark.parametrize(
    "inputs",
    [
        {"prompt": "product hero image"},
        {"prompt": "product hero image", "output_path": "generated_image.png"},
        {"prompt": "product hero image", "output_path": "/tmp/generated_image.png"},
    ],
)
def test_openai_image_requires_project_output_path_before_client_creation(
    inputs: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls: list[str] = []

    class FakeOpenAI:
        def __init__(self) -> None:
            calls.append("client")
            raise AssertionError("OpenAI client should not be created for invalid output_path")

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    result = OpenAIImage().execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert calls == []


@pytest.mark.parametrize(
    ("tool", "env_var"),
    [
        (FluxImage(), "FAL_KEY"),
        (RecraftImage(), "FAL_KEY"),
        (GoogleImagen(), "GOOGLE_API_KEY"),
    ],
)
def test_api_image_generators_require_project_output_path_before_requests(
    tool: Any,
    env_var: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(env_var, "test-key")
    calls: list[object] = []

    def fake_post(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        raise AssertionError("network should not be called without project output_path")

    monkeypatch.setattr("requests.post", fake_post)

    result = tool.execute({"prompt": "product hero image"})

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert calls == []


def test_diagram_gen_mermaid_cli_fails_when_expected_output_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    output_path = Path(_project_graphics_output("diagram.png"))
    tool = DiagramGen()

    monkeypatch.setattr(DiagramGen, "_has_mermaid", lambda self: True)
    monkeypatch.setattr(DiagramGen, "run_command", lambda *args, **kwargs: object())

    result = tool.execute(
        {
            "diagram_type": "mermaid",
            "definition": "graph TD\n  A[Start] --> B[End]",
            "output_path": _project_graphics_output("diagram.png"),
        }
    )

    assert result.success is False
    assert output_path.as_posix() in (result.error or "")


def test_diagram_gen_rejects_non_finite_mermaid_config_before_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    output_path = tmp_path / _project_graphics_output("diagram.png")
    run_calls: list[object] = []

    def fake_run_command(self, cmd, *args, **kwargs):
        run_calls.append(cmd)
        output_path.write_bytes(b"png")

    monkeypatch.setattr(DiagramGen, "_has_mermaid", lambda self: True)
    monkeypatch.setattr(DiagramGen, "run_command", fake_run_command)

    result = DiagramGen().execute(
        {
            "diagram_type": "mermaid",
            "definition": "graph TD\n  A[Start] --> B[End]",
            "theme": math.nan,
            "output_path": _project_graphics_output("diagram.png"),
        }
    )

    assert result.success is False
    assert "strict JSON" in (result.error or "")
    assert run_calls == []
    assert not output_path.exists()
    assert not output_path.with_suffix(".mmd").exists()
    assert not output_path.with_suffix(".mermaid.json").exists()


@pytest.mark.parametrize("output_kind", ["missing", "relative", "absolute"])
def test_math_animate_requires_project_output_path_before_manim(
    output_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    run_calls: list[list[str]] = []

    def fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        run_calls.append(cmd)
        media_dir = Path(kwargs["cwd"]) / "media" / "videos" / "scene" / "480p15"
        media_dir.mkdir(parents=True)
        (media_dir / "Demo.mp4").write_bytes(b"mp4")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("tools.graphics.math_animate.shutil.which", lambda _cmd: "/usr/bin/manim")
    monkeypatch.setattr("tools.graphics.math_animate.subprocess.run", fake_run)

    inputs: dict[str, object] = {
        "scene_code": "class Demo(Scene):\n    def construct(self):\n        pass",
        "scene_name": "Demo",
    }
    if output_kind == "relative":
        inputs["output_path"] = "math.mp4"
        forbidden_output = tmp_path / "math.mp4"
    elif output_kind == "absolute":
        forbidden_output = tmp_path / "math.mp4"
        inputs["output_path"] = str(forbidden_output)
    else:
        forbidden_output = tmp_path / "manim_Demo.mp4"

    result = MathAnimate().execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert run_calls == []
    assert not forbidden_output.exists()


def test_math_animate_schema_requires_output_path() -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {"scene_code": "class Demo(Scene):\n    def construct(self):\n        pass"},
            MathAnimate.input_schema,
        )


def test_math_animate_success_payload_includes_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    output_path = _project_graphics_output("math.mp4")

    def fake_which(command: str) -> str | None:
        if command == "manim":
            return "/usr/bin/manim"
        return None

    def fake_run(
        cmd: list[str],
        *args: Any,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        media_dir = Path(kwargs["cwd"]) / "media" / "videos" / "scene" / "480p15"
        media_dir.mkdir(parents=True)
        (media_dir / "Demo.mp4").write_bytes(b"mp4")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("tools.graphics.math_animate.shutil.which", fake_which)
    monkeypatch.setattr("tools.graphics.math_animate.subprocess.run", fake_run)

    result = MathAnimate().execute(
        {
            "scene_code": "class Demo(Scene):\n    def construct(self):\n        pass",
            "scene_name": "Demo",
            "quality": "low",
            "output_path": output_path,
        }
    )

    assert result.success is True
    assert result.data["output_path"] == output_path
    assert result.artifacts == [output_path]
    assert (tmp_path / output_path).exists()


def test_math_animate_success_payload_matches_output_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    output_path = _project_graphics_output("math-schema.mp4")

    def fake_which(command: str) -> str | None:
        if command == "manim":
            return "/usr/bin/manim"
        return None

    def fake_run(
        cmd: list[str],
        *args: Any,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        media_dir = Path(kwargs["cwd"]) / "media" / "videos" / "scene" / "480p15"
        media_dir.mkdir(parents=True)
        (media_dir / "Demo.mp4").write_bytes(b"mp4")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("tools.graphics.math_animate.shutil.which", fake_which)
    monkeypatch.setattr("tools.graphics.math_animate.subprocess.run", fake_run)

    output_properties = MathAnimate.output_schema["properties"]
    assert {
        "scene_name",
        "quality",
        "format",
        "output",
        "output_path",
        "resolution",
        "fps",
        "file_size_bytes",
    } <= set(output_properties)

    result = MathAnimate().execute(
        {
            "scene_code": "class Demo(Scene):\n    def construct(self):\n        pass",
            "scene_name": "Demo",
            "quality": "low",
            "output_path": output_path,
        }
    )

    assert result.success is True
    assert result.artifacts == [output_path]
    jsonschema.validate(instance=result.data, schema=MathAnimate.output_schema)


def test_screen_capture_selector_status_requires_installed_capture_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingFFmpegProvider:
        def get_status(self) -> ToolStatus:
            return ToolStatus.UNAVAILABLE

    class MissingCapProvider:
        def get_status(self) -> ToolStatus:
            return ToolStatus.AVAILABLE

        def execute(self, inputs: dict[str, object]) -> Any:
            return type(
                "Result",
                (),
                {"success": True, "data": {"installed": False, "running": False}},
            )()

    monkeypatch.setattr(
        ScreenCaptureSelector,
        "_providers",
        lambda self: {"ffmpeg": MissingFFmpegProvider(), "cap": MissingCapProvider()},
    )

    assert ScreenCaptureSelector().get_status() == ToolStatus.UNAVAILABLE


@pytest.mark.parametrize(
    "inputs",
    [
        {"operation": "record", "preferred_provider": "ffmpeg"},
        {
            "operation": "record",
            "preferred_provider": "ffmpeg",
            "output_path": "recording.mp4",
        },
        {
            "operation": "record",
            "preferred_provider": "cap",
            "output_path": "tmp/recording.mp4",
        },
    ],
)
def test_screen_capture_selector_record_requires_project_output_path_before_provider_call(
    inputs: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class Provider:
        def get_status(self) -> ToolStatus:
            return ToolStatus.AVAILABLE

        def execute(self, provider_inputs: dict[str, object]) -> Any:
            calls.append(provider_inputs)
            return SimpleNamespace(success=True, data={}, artifacts=[])

    monkeypatch.setattr(
        ScreenCaptureSelector,
        "_providers",
        lambda self: {"ffmpeg": Provider(), "cap": Provider()},
    )

    result = ScreenCaptureSelector().execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert calls == []


def test_screen_capture_selector_record_passes_valid_project_output_to_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FFmpegProvider:
        def get_status(self) -> ToolStatus:
            return ToolStatus.AVAILABLE

        def execute(self, provider_inputs: dict[str, object]) -> Any:
            calls.append(provider_inputs)
            return ToolResult(
                success=True,
                data={"output_path": provider_inputs["output_path"]},
                artifacts=[provider_inputs["output_path"]],
            )

    monkeypatch.setattr(
        ScreenCaptureSelector,
        "_providers",
        lambda self: {"ffmpeg": FFmpegProvider()},
    )

    result = ScreenCaptureSelector().execute(
        {
            "operation": "record",
            "preferred_provider": "ffmpeg",
            "output_path": "projects/test-screen-demo/renders/recording.mp4",
            "duration_seconds": 5,
            "fps": 24,
            "capture_audio": False,
        }
    )

    assert result.success is True
    assert calls == [
        {
            "output_path": str(
                Path("projects/test-screen-demo/renders/recording.mp4")
            ),
            "duration_seconds": 5,
            "fps": 24,
            "capture_audio": False,
            "region": None,
        }
    ]


@pytest.mark.parametrize(
    "inputs",
    [
        {"operation": "pick_latest"},
        {"operation": "pick_latest", "output_path": "recording.mp4"},
        {"operation": "pick_latest", "output_path": "/tmp/recording.mp4"},
    ],
)
def test_screen_capture_selector_pick_latest_requires_project_output_path_before_provider_call(
    inputs: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class CapProvider:
        def execute(self, provider_inputs: dict[str, object]) -> Any:
            calls.append(provider_inputs)
            return SimpleNamespace(success=True, data={}, artifacts=[])

    monkeypatch.setattr(
        ScreenCaptureSelector,
        "_providers",
        lambda self: {"cap": CapProvider()},
    )

    result = ScreenCaptureSelector().execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert calls == []


def test_screen_capture_selector_pick_latest_copies_cap_recording_to_project_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class CapProvider:
        def execute(self, provider_inputs: dict[str, object]) -> Any:
            calls.append(provider_inputs)
            if provider_inputs["operation"] == "pick_latest":
                return ToolResult(
                    success=True,
                    data={
                        "output_path": provider_inputs["output_dir"],
                        "capture_method": "cap",
                    },
                    artifacts=[provider_inputs["output_dir"]],
                )
            return ToolResult(
                success=True,
                data={
                    "recordings": [
                        {
                            "path": "/home/user/Cap/external.mp4",
                            "size_mb": 1.0,
                        }
                    ]
                },
                artifacts=[],
            )

    monkeypatch.setattr(
        ScreenCaptureSelector,
        "_providers",
        lambda self: {"cap": CapProvider()},
    )

    result = ScreenCaptureSelector().execute(
        {
            "operation": "pick_latest",
            "output_path": "projects/test-screen-demo/renders/latest.mp4",
            "since_minutes": 9,
        }
    )

    assert result.success is True
    assert calls == [
        {
            "operation": "pick_latest",
            "output_dir": str(Path("projects/test-screen-demo/renders/latest.mp4")),
            "since_minutes": 9,
        }
    ]
    assert result.data["output_path"] == "projects/test-screen-demo/renders/latest.mp4"
    assert result.artifacts == ["projects/test-screen-demo/renders/latest.mp4"]


def test_screen_capture_selector_schema_requires_output_path_for_record_and_pick_latest() -> None:
    schema = ScreenCaptureSelector().input_schema

    jsonschema.validate({"operation": "recommend"}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"operation": "record", "preferred_provider": "ffmpeg"}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"operation": "pick_latest"}, schema)


def test_screen_capture_selector_idempotency_key_includes_routed_record_inputs() -> None:
    tool = ScreenCaptureSelector()
    base = {
        "operation": "record",
        "preferred_provider": "ffmpeg",
        "output_path": "recording-a.mp4",
        "duration_seconds": 30,
        "fps": 30,
        "capture_audio": True,
        "region": {"x": 0, "y": 0, "width": 1280, "height": 720},
    }

    assert tool.idempotency_key(base) != tool.idempotency_key(
        {**base, "output_path": "recording-b.mp4"}
    )


def test_math_animate_preview_quality_defaults_to_gif_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_which(command: str) -> str | None:
        if command == "manim":
            return "/usr/bin/manim"
        return None

    def fake_run(
        cmd: list[str],
        *args: Any,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        media_dir = Path(kwargs["cwd"]) / "media" / "videos" / "scene" / "480p15"
        media_dir.mkdir(parents=True)
        (media_dir / "Demo.gif").write_bytes(b"gif89a")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("tools.graphics.math_animate.shutil.which", fake_which)
    monkeypatch.setattr("tools.graphics.math_animate.subprocess.run", fake_run)

    result = MathAnimate().execute(
        {
            "scene_code": "class Demo(Scene):\n    def construct(self):\n        pass",
            "quality": "preview",
            "output_path": _project_graphics_output("preview.gif"),
        }
    )

    assert result.success, result.error
    assert result.data["format"] == "gif"
    assert Path(result.data["output"]).suffix == ".gif"
    assert (tmp_path / result.data["output"]).exists()
