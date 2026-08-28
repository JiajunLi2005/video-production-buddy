from __future__ import annotations

import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from styles.playbook_loader import load_playbook
from tools.validation.scene_fidelity_check import check_plan, load_registry
from tools.video.remotion_caption_burn import RemotionCaptionBurn
from tools.video.video_compose import VideoCompose


@pytest.fixture
def project_renders_dir(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    project_dir = repo_root / "projects" / f"pytest-video-compose-{tmp_path.name}"
    shutil.rmtree(project_dir, ignore_errors=True)
    renders_dir = project_dir / "renders"
    yield renders_dir
    shutil.rmtree(project_dir, ignore_errors=True)


def _assert_video_compose_output_schema_matches(
    payload: dict[str, object],
    expected_properties: set[str],
) -> None:
    output_properties = VideoCompose.output_schema["properties"]
    assert expected_properties <= set(output_properties)
    jsonschema.validate(instance=payload, schema=VideoCompose.output_schema)


def test_ffmpeg_compose_uses_source_in_seconds_for_source_seek(
    monkeypatch,
    tmp_path,
    project_renders_dir,
):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"stub-video")
    output = project_renders_dir / "out.mp4"
    commands: list[list[str]] = []

    composer = VideoCompose()
    monkeypatch.setattr(composer, "_has_audio_stream", lambda _path: False)

    def fake_run_command(cmd, *args, **kwargs):
        commands.append(list(cmd))
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"stub-output")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(composer, "run_command", fake_run_command)

    result = composer.execute(
        {
            "operation": "compose",
            "edit_decisions": {
                "version": "1.0",
                "render_runtime": "ffmpeg",
                "cuts": [
                    {
                        "id": "cut-1",
                        "source": str(source),
                        "in_seconds": 12.0,
                        "out_seconds": 14.5,
                        "source_in_seconds": 1.25,
                    }
                ],
            },
            "output_path": str(output),
        }
    )

    assert result.success, result.error
    assert result.data is not None
    assert result.data["output_path"] == output.as_posix()
    trim_cmd = commands[0]
    assert trim_cmd[trim_cmd.index("-ss") + 1] == "1.25"
    assert trim_cmd[trim_cmd.index("-t") + 1] == "2.5"


def test_ffmpeg_compose_escapes_single_quotes_in_concat_list(
    monkeypatch,
    tmp_path,
    project_renders_dir,
):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"stub-video")
    quoted_dir = project_renders_dir / "work'space"
    output = quoted_dir / "out.mp4"
    concat_list_body: dict[str, str] = {}

    composer = VideoCompose()
    monkeypatch.setattr(composer, "_has_audio_stream", lambda _path: False)

    def fake_run_command(cmd, *args, **kwargs):
        if "-f" in cmd and "concat" in cmd:
            list_path = Path(cmd[cmd.index("-i") + 1])
            concat_list_body["body"] = list_path.read_text(encoding="utf-8")
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"stub-output")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(composer, "run_command", fake_run_command)

    result = composer.execute(
        {
            "operation": "compose",
            "edit_decisions": {
                "version": "1.0",
                "render_runtime": "ffmpeg",
                "cuts": [
                    {
                        "id": "cut-1",
                        "source": str(source),
                        "in_seconds": 0.0,
                        "out_seconds": 1.0,
                    }
                ],
            },
            "output_path": str(output),
        }
    )

    assert result.success, result.error
    assert "work'\\''space" in concat_list_body["body"]


def test_ffmpeg_compose_escapes_single_quotes_in_subtitle_filter(
    monkeypatch,
    tmp_path,
    project_renders_dir,
):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"stub-video")
    subtitle_dir = tmp_path / "subtitle's"
    subtitle_dir.mkdir()
    subtitle = subtitle_dir / "subs.ass"
    subtitle.write_text("[Script Info]\n", encoding="utf-8")
    output = project_renders_dir / "out.mp4"
    captured_filters: list[str] = []

    composer = VideoCompose()
    monkeypatch.setattr(composer, "_has_audio_stream", lambda _path: False)

    def fake_run_command(cmd, *args, **kwargs):
        if "-vf" in cmd:
            captured_filters.append(cmd[cmd.index("-vf") + 1])
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"stub-output")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(composer, "run_command", fake_run_command)

    result = composer.execute(
        {
            "operation": "compose",
            "subtitle_path": str(subtitle),
            "edit_decisions": {
                "version": "1.0",
                "render_runtime": "ffmpeg",
                "cuts": [
                    {
                        "id": "cut-1",
                        "source": str(source),
                        "in_seconds": 0.0,
                        "out_seconds": 1.0,
                    }
                ],
            },
            "output_path": str(output),
        }
    )

    assert result.success, result.error
    assert any("subtitle\\'s" in vf for vf in captured_filters)


def test_burn_subtitles_escapes_single_quotes_in_subtitle_filter(
    monkeypatch,
    tmp_path,
    project_renders_dir,
):
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"stub-video")
    subtitle_dir = tmp_path / "subtitle's"
    subtitle_dir.mkdir()
    subtitle = subtitle_dir / "subs.ass"
    subtitle.write_text("[Script Info]\n", encoding="utf-8")
    output_path = project_renders_dir / "out.mp4"
    captured: dict[str, str] = {}
    composer = VideoCompose()

    def fake_run_command(cmd, *args, **kwargs):
        captured["vf"] = cmd[cmd.index("-vf") + 1]
        output_path.write_bytes(b"subtitled")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(composer, "run_command", fake_run_command)

    result = composer.execute(
        {
            "operation": "burn_subtitles",
            "input_path": str(input_path),
            "subtitle_path": str(subtitle),
            "output_path": str(output_path),
        }
    )

    assert result.success, result.error
    assert result.data is not None
    assert result.data["output_path"] == output_path.as_posix()
    assert "subtitle\\'s" in captured["vf"]


@pytest.mark.parametrize(
    ("operation", "output_kind"),
    [
        ("burn_subtitles", "missing"),
        ("burn_subtitles", "relative"),
        ("burn_subtitles", "absolute"),
        ("overlay", "missing"),
        ("overlay", "relative"),
        ("overlay", "absolute"),
        ("encode", "missing"),
        ("encode", "relative"),
        ("encode", "absolute"),
    ],
)
def test_video_compose_postprocess_requires_project_output_path_before_ffmpeg(
    operation: str,
    output_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    subtitle_path = tmp_path / "subs.ass"
    subtitle_path.write_text("[Script Info]\n", encoding="utf-8")
    overlay_path = tmp_path / "overlay.png"
    overlay_path.write_bytes(b"png")
    commands: list[list[str]] = []

    def fake_run_command(cmd: list[str], *args: object, **kwargs: object) -> object:
        commands.append(cmd)
        Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[-1]).write_bytes(b"video")
        return SimpleNamespace(stdout="")

    composer = VideoCompose()
    monkeypatch.setattr(composer, "run_command", fake_run_command)
    inputs: dict[str, object] = {
        "operation": operation,
        "input_path": str(input_path),
    }
    if operation == "burn_subtitles":
        inputs["subtitle_path"] = str(subtitle_path)
        default_output = input_path.with_stem(f"{input_path.stem}_subtitled")
    elif operation == "overlay":
        inputs["overlays"] = [{"asset_path": str(overlay_path), "x": 0, "y": 0}]
        default_output = input_path.with_stem(f"{input_path.stem}_overlay")
    else:
        default_output = input_path.with_stem(f"{input_path.stem}_encoded")

    if output_kind == "relative":
        inputs["output_path"] = f"{operation}.mp4"
        forbidden_output = tmp_path / f"{operation}.mp4"
    elif output_kind == "absolute":
        forbidden_output = tmp_path / f"{operation}.mp4"
        inputs["output_path"] = str(forbidden_output)
    else:
        forbidden_output = default_output

    result = composer.execute(inputs)

    assert result.success is False
    assert "output_path" in (result.error or "")
    assert "projects/<project-name>/" in (result.error or "")
    assert commands == []
    assert not forbidden_output.exists()


@pytest.mark.parametrize(
    "operation",
    ["compose", "render", "remotion_render", "burn_subtitles", "overlay", "encode"],
)
def test_video_compose_schema_requires_output_path(operation: str) -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            instance={"operation": operation},
            schema=VideoCompose.input_schema,
        )


def test_video_compose_idempotency_key_includes_output_and_render_inputs():
    composer = VideoCompose()
    base = {
        "operation": "compose",
        "input_path": "input.mp4",
        "output_path": "out-a.mp4",
        "edit_decisions": {"cuts": [{"source": "a.mp4", "in_seconds": 0, "out_seconds": 1}]},
        "asset_manifest": {"assets": [{"id": "a", "path": "a.mp4"}]},
        "audio_path": "mix-a.wav",
        "subtitle_path": "subs-a.ass",
        "subtitle_style": {"font_size": 24},
        "profile": "youtube_landscape",
        "options": {"subtitle_burn": True},
        "codec": "libx264",
        "crf": 23,
        "preset": "medium",
        "script_path": "scripts/narration-a.txt",
        "script_text": "Narration A",
        "narration_transcript_path": "transcripts/narration-a.json",
    }
    variants = [
        {"output_path": "out-b.mp4"},
        {"asset_manifest": {"assets": [{"id": "a", "path": "other.mp4"}]}},
        {"audio_path": "mix-b.wav"},
        {"subtitle_path": "subs-b.ass"},
        {"subtitle_style": {"font_size": 30}},
        {"profile": "tiktok"},
        {"options": {"subtitle_burn": False}},
        {"codec": "libx265"},
        {"crf": 18},
        {"preset": "slow"},
        {"script_path": "scripts/narration-b.txt"},
        {"script_text": "Narration B"},
        {"narration_transcript_path": "transcripts/narration-b.json"},
    ]

    base_key = composer.idempotency_key(base)

    for variant in variants:
        assert composer.idempotency_key({**base, **variant}) != base_key


def test_ffmpeg_compose_rejects_unknown_media_profile(
    monkeypatch,
    tmp_path,
    project_renders_dir,
):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"stub-video")
    composer = VideoCompose()
    monkeypatch.setattr(composer, "_has_audio_stream", lambda _path: False)

    def fake_run_command(cmd, *args, **kwargs):
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"stub-output")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(composer, "run_command", fake_run_command)

    result = composer.execute(
        {
            "operation": "compose",
            "profile": "not-a-real-profile",
            "edit_decisions": {
                "version": "1.0",
                "render_runtime": "ffmpeg",
                "cuts": [
                    {
                        "id": "cut-1",
                        "source": str(source),
                        "in_seconds": 0.0,
                        "out_seconds": 1.0,
                    }
                ],
            },
            "output_path": str(project_renders_dir / "out.mp4"),
        }
    )

    assert not result.success
    assert "Unknown profile" in (result.error or "")


def test_video_compose_encode_rejects_unknown_media_profile(
    monkeypatch,
    tmp_path,
    project_renders_dir,
):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"stub-video")
    composer = VideoCompose()
    monkeypatch.setattr(composer, "run_command", lambda *args, **kwargs: None)

    result = composer.execute(
        {
            "operation": "encode",
            "input_path": str(source),
            "profile": "not-a-real-profile",
            "output_path": str(project_renders_dir / "encoded.mp4"),
        }
    )

    assert not result.success
    assert "Unknown profile" in (result.error or "")


def test_remotion_failure_guidance_uses_pnpm_lockfile(
    monkeypatch,
    project_renders_dir,
):
    composer = VideoCompose()
    monkeypatch.setattr(composer, "_remotion_available", lambda: True)
    monkeypatch.setattr(composer, "_pre_compose_validation", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        composer,
        "_remotion_render",
        lambda _inputs: SimpleNamespace(success=False, error="render failed"),
    )

    result = composer.execute(
        {
            "operation": "render",
            "edit_decisions": {
                "version": "1.0",
                "renderer_family": "explainer-data",
                "render_runtime": "remotion",
                "cuts": [
                    {
                        "id": "cut-1",
                        "type": "text_card",
                        "source": "remotion:text_card",
                        "text": "Locked Remotion",
                        "in_seconds": 0,
                        "out_seconds": 1,
                    }
                ],
            },
            "asset_manifest": {"version": "1.0", "assets": []},
            "output_path": str(project_renders_dir / "out.mp4"),
        }
    )

    assert not result.success
    assert "pnpm install --frozen-lockfile" in (result.error or "")
    assert "&& npm install" not in (result.error or "")


def test_remotion_props_reject_non_finite_values_before_render(
    monkeypatch,
    project_renders_dir,
):
    composer = VideoCompose()
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/npx")
    output = project_renders_dir / "out.mp4"
    render_called = False

    def fake_run_command(*args, **kwargs):
        nonlocal render_called
        render_called = True
        output.write_bytes(b"rendered")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(composer, "run_command", fake_run_command)

    result = composer.execute(
        {
            "operation": "remotion_render",
            "composition_data": {
                "version": "1.0",
                "renderer_family": "explainer-data",
                "render_runtime": "remotion",
                "metadata": {"confidence": math.nan},
                "cuts": [
                    {
                        "id": "cut-1",
                        "type": "text_card",
                        "source": "remotion:text_card",
                        "text": "Strict props",
                        "in_seconds": 0,
                        "out_seconds": 1,
                    }
                ],
            },
            "output_path": str(output),
        }
    )

    assert not result.success
    assert "strict JSON" in result.error
    assert render_called is False
    assert not (output.parent / ".remotion_props.json").exists()
    assert not output.exists()


def test_video_compose_remotion_render_success_payload_includes_output_path(
    monkeypatch,
    project_renders_dir,
):
    composer = VideoCompose()
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/npx")
    output = project_renders_dir / "out.mp4"

    def fake_run_command(cmd, *args, **kwargs):
        output.write_bytes(b"rendered")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(composer, "run_command", fake_run_command)

    result = composer.execute(
        {
            "operation": "remotion_render",
            "composition_data": {
                "version": "1.0",
                "renderer_family": "explainer-data",
                "render_runtime": "remotion",
                "cuts": [
                    {
                        "id": "cut-1",
                        "type": "text_card",
                        "source": "remotion:text_card",
                        "text": "Render me",
                        "in_seconds": 0,
                        "out_seconds": 1,
                    }
                ],
            },
            "output_path": str(output),
        }
    )

    assert result.success is True
    assert result.data is not None
    assert result.data["output"] == output.as_posix()
    assert result.data["output_path"] == output.as_posix()
    assert result.artifacts == [output.as_posix()]
    _assert_video_compose_output_schema_matches(
        result.data,
        {"operation", "output", "output_path", "profile"},
    )


def test_video_compose_overlay_success_payload_includes_output_path(
    monkeypatch,
    tmp_path,
    project_renders_dir,
):
    input_path = tmp_path / "input.mp4"
    overlay_path = tmp_path / "overlay.png"
    input_path.write_bytes(b"input")
    overlay_path.write_bytes(b"overlay")
    output = project_renders_dir / "overlay.mp4"
    composer = VideoCompose()

    def fake_run_command(cmd, *args, **kwargs):
        output.write_bytes(b"overlayed")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(composer, "run_command", fake_run_command)

    result = composer.execute(
        {
            "operation": "overlay",
            "input_path": str(input_path),
            "overlays": [{"asset_path": str(overlay_path), "x": 0, "y": 0}],
            "output_path": str(output),
        }
    )

    assert result.success is True
    assert result.data is not None
    assert result.data["output"] == output.as_posix()
    assert result.data["output_path"] == output.as_posix()
    assert result.artifacts == [output.as_posix()]
    _assert_video_compose_output_schema_matches(
        result.data,
        {"operation", "overlay_count", "output", "output_path"},
    )


def test_video_compose_encode_success_payload_includes_output_path(
    monkeypatch,
    tmp_path,
    project_renders_dir,
):
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"input")
    output = project_renders_dir / "encoded.mp4"
    composer = VideoCompose()

    def fake_run_command(cmd, *args, **kwargs):
        output.write_bytes(b"encoded")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(composer, "run_command", fake_run_command)

    result = composer.execute(
        {
            "operation": "encode",
            "input_path": str(input_path),
            "output_path": str(output),
        }
    )

    assert result.success is True
    assert result.data is not None
    assert result.data["output"] == output.as_posix()
    assert result.data["output_path"] == output.as_posix()
    assert result.artifacts == [output.as_posix()]
    _assert_video_compose_output_schema_matches(
        result.data,
        {"operation", "codec", "crf", "profile", "output", "output_path"},
    )


def test_video_compose_coerced_artifact_path_rejects_non_strict_json(tmp_path: Path):
    artifact_path = tmp_path / "edit_decisions.json"
    artifact_path.write_text('{"render_runtime": NaN, "cuts": []}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="strict JSON"):
        VideoCompose._coerce_artifact(str(artifact_path), "edit_decisions")


def test_remotion_caption_burn_install_instructions_use_pnpm_lockfile():
    instructions = RemotionCaptionBurn.install_instructions

    assert "pnpm install --frozen-lockfile" in instructions
    assert "npm install in remotion-composer" not in instructions


def test_render_output_path_rejects_project_artifacts_dir():
    composer = VideoCompose()
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "projects" / "contract-test" / "artifacts" / "final.mp4"

    returned_path, error = composer._required_render_output_path(
        {"output_path": str(output_path)},
        "render",
    )

    assert returned_path is None
    assert error is not None
    assert not error.success
    assert "projects/<project-name>/renders/" in (error.error or "")


def test_render_output_path_rejects_parent_traversal_between_projects():
    composer = VideoCompose()

    returned_path, error = composer._required_render_output_path(
        {"output_path": "projects/contract-test/../other/renders/final.mp4"},
        "render",
    )

    assert returned_path is None
    assert error is not None
    assert not error.success
    assert "projects/<project-name>/renders/" in (error.error or "")


@pytest.mark.parametrize("output_path", [[], {}, 123])
def test_render_output_path_rejects_non_string_values(output_path: object):
    composer = VideoCompose()

    returned_path, error = composer._required_render_output_path(
        {"output_path": output_path},
        "render",
    )

    assert returned_path is None
    assert error is not None
    assert not error.success
    assert "output_path for render must be a string path" in (error.error or "")


@pytest.mark.parametrize(
    "output_path",
    [
        " projects/contract-test/renders/final.mp4",
        "projects/contract-test/renders/final.mp4 ",
    ],
)
def test_render_output_path_rejects_padded_project_paths(output_path: str):
    composer = VideoCompose()

    returned_path, error = composer._required_render_output_path(
        {"output_path": output_path},
        "render",
    )

    assert returned_path is None
    assert error is not None
    assert not error.success
    assert "projects/<project-name>/renders/" in (error.error or "")


def test_render_output_path_requires_file_extension():
    composer = VideoCompose()

    returned_path, error = composer._required_render_output_path(
        {"output_path": "projects/contract-test/renders/final"},
        "render",
    )

    assert returned_path is None
    assert error is not None
    assert not error.success
    assert "projects/<project-name>/renders/<file>.mp4" in (error.error or "")


def test_render_output_path_accepts_project_renders_dir():
    composer = VideoCompose()
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "projects" / "contract-test" / "renders" / "final.mp4"

    returned_path, error = composer._required_render_output_path(
        {"output_path": str(output_path)},
        "render",
    )

    assert error is None
    assert returned_path == output_path


def test_render_output_path_rejects_absolute_path_outside_project_renders_dir(tmp_path):
    composer = VideoCompose()
    output_path = tmp_path / "outside-project.mp4"

    returned_path, error = composer._required_render_output_path(
        {"output_path": str(output_path)},
        "render",
    )

    assert returned_path is None
    assert error is not None
    assert not error.success
    assert "projects/<project-name>/renders/" in (error.error or "")


def test_ffmpeg_compose_requires_project_renders_output_path_before_commands(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"stub-video")
    composer = VideoCompose()
    commands: list[list[str]] = []
    monkeypatch.setattr(composer, "_has_audio_stream", lambda _path: False)

    def fake_run_command(cmd, *args, **kwargs):
        commands.append(list(cmd))
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"stub-output")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(composer, "run_command", fake_run_command)

    result = composer.execute(
        {
            "operation": "compose",
            "edit_decisions": {
                "version": "1.0",
                "render_runtime": "ffmpeg",
                "cuts": [
                    {
                        "id": "cut-1",
                        "source": str(source),
                        "in_seconds": 0.0,
                        "out_seconds": 1.0,
                    }
                ],
            },
        }
    )

    assert not result.success
    assert "output_path required for compose" in (result.error or "")
    assert commands == []
    assert not (Path.cwd() / "composed_output.mp4").exists()


def test_ffmpeg_compose_normalizes_segments_to_selected_profile(
    monkeypatch,
    tmp_path,
    project_renders_dir,
):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"stub-video")
    output = project_renders_dir / "out.mp4"
    commands: list[list[str]] = []

    composer = VideoCompose()
    monkeypatch.setattr(composer, "_has_audio_stream", lambda _path: False)

    def fake_run_command(cmd, *args, **kwargs):
        commands.append(list(cmd))
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"stub-output")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(composer, "run_command", fake_run_command)

    result = composer.execute(
        {
            "operation": "compose",
            "profile": "tiktok",
            "edit_decisions": {
                "version": "1.0",
                "render_runtime": "ffmpeg",
                "cuts": [
                    {
                        "id": "cut-1",
                        "source": str(source),
                        "in_seconds": 0.0,
                        "out_seconds": 2.0,
                    }
                ],
            },
            "output_path": str(output),
        }
    )

    assert result.success, result.error
    trim_cmd = commands[0]
    vf = trim_cmd[trim_cmd.index("-filter:v") + 1]
    assert "scale=1080:1920:force_original_aspect_ratio=decrease" in vf
    assert "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black" in vf
    assert "fps=30" in vf


def test_ffmpeg_compose_does_not_burn_disabled_subtitles(
    monkeypatch,
    tmp_path,
    project_renders_dir,
):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"stub-video")
    subtitle = tmp_path / "subtitles.ass"
    subtitle.write_text(
        "[Script Info]\nPlayResX: 1920\nPlayResY: 1080\n", encoding="utf-8"
    )
    output = project_renders_dir / "out.mp4"
    commands: list[list[str]] = []

    composer = VideoCompose()
    monkeypatch.setattr(composer, "_has_audio_stream", lambda _path: False)

    def fake_run_command(cmd, *args, **kwargs):
        commands.append(list(cmd))
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"stub-output")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(composer, "run_command", fake_run_command)

    result = composer.execute(
        {
            "operation": "compose",
            "edit_decisions": {
                "version": "1.0",
                "render_runtime": "ffmpeg",
                "cuts": [
                    {
                        "id": "cut-1",
                        "source": str(source),
                        "in_seconds": 0.0,
                        "out_seconds": 2.0,
                    }
                ],
                "subtitles": {"enabled": False, "source": str(subtitle)},
            },
            "output_path": str(output),
        }
    )

    assert result.success, result.error
    assert not any(
        "subtitles=" in str(part) for command in commands for part in command
    )


def test_ffmpeg_render_resolves_subtitle_asset_id_before_compose(
    monkeypatch,
    tmp_path,
    project_renders_dir,
):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"stub-video")
    subtitle = tmp_path / "subtitles.ass"
    subtitle.write_text(
        "[Script Info]\nPlayResX: 1920\nPlayResY: 1080\n", encoding="utf-8"
    )
    output = project_renders_dir / "out.mp4"
    captured: dict[str, object] = {}

    composer = VideoCompose()
    monkeypatch.setattr(composer, "_pre_compose_validation", lambda *args, **kwargs: None)

    def fake_compose(inputs):
        captured.update(inputs)
        return SimpleNamespace(success=True, error=None, data={}, artifacts=[])

    monkeypatch.setattr(composer, "_compose", fake_compose)

    result = composer.execute(
        {
            "operation": "render",
            "edit_decisions": {
                "version": "1.0",
                "renderer_family": "cinematic-trailer",
                "render_runtime": "ffmpeg",
                "cuts": [
                    {
                        "id": "cut-1",
                        "source": "video-1",
                        "in_seconds": 0.0,
                        "out_seconds": 2.0,
                    }
                ],
                "subtitles": {"enabled": True, "source": "subtitle-1"},
            },
            "asset_manifest": {
                "version": "1.0",
                "assets": [
                    {
                        "id": "video-1",
                        "type": "video",
                        "path": str(source),
                        "source_tool": "fixture",
                        "scene_id": "scene-1",
                    },
                    {
                        "id": "subtitle-1",
                        "type": "subtitle",
                        "path": str(subtitle),
                        "source_tool": "subtitle_gen",
                        "scene_id": "global",
                    },
                ],
            },
            "output_path": str(output),
        }
    )

    assert result.success, result.error
    assert captured["subtitle_path"] == str(subtitle)


def test_scene_fidelity_accepts_plain_media_cut_without_component_type():
    registry = load_registry()

    report = check_plan(
        {
            "version": "1.0",
            "render_runtime": "ffmpeg",
            "cuts": [
                {
                    "id": "cut-1",
                    "source": "projects/example/assets/video/clip.mp4",
                    "in_seconds": 0,
                    "out_seconds": 2,
                    "maps_to_beat": "hook",
                }
            ],
        },
        registry,
    )

    assert report["ok"], report["issues"]


def test_scene_fidelity_blocks_overlapping_source_reuse():
    registry = load_registry()

    report = check_plan(
        {
            "version": "1.0",
            "render_runtime": "ffmpeg",
            "cuts": [
                {
                    "id": "cut-1",
                    "source": "projects/example/assets/video/clip.mp4",
                    "in_seconds": 0.0,
                    "out_seconds": 2.0,
                    "source_in_seconds": 0.0,
                    "maps_to_beat": "hook",
                },
                {
                    "id": "cut-2",
                    "source": "projects/example/assets/video/clip.mp4",
                    "in_seconds": 2.0,
                    "out_seconds": 4.0,
                    "source_in_seconds": 1.0,
                    "maps_to_beat": "build",
                },
            ],
        },
        registry,
    )

    assert not report["ok"]
    assert report["issues"][0]["kind"] == "overlapping_source_reuse"
    assert report["issues"][0]["scene_id"] == "cut-2"


def test_scene_fidelity_blocks_remotion_source_reuse_without_source_in_seconds():
    registry = load_registry()

    report = check_plan(
        {
            "version": "1.0",
            "render_runtime": "remotion",
            "cuts": [
                {
                    "id": "cut-1",
                    "source": "projects/example/assets/video/clip.mp4",
                    "in_seconds": 0.0,
                    "out_seconds": 2.0,
                    "maps_to_beat": "hook",
                },
                {
                    "id": "cut-2",
                    "source": "projects/example/assets/video/clip.mp4",
                    "in_seconds": 2.0,
                    "out_seconds": 4.0,
                    "maps_to_beat": "build",
                },
            ],
        },
        registry,
    )

    assert not report["ok"]
    assert report["issues"][0]["kind"] == "overlapping_source_reuse"
    assert report["issues"][0]["scene_id"] == "cut-2"
    assert report["issues"][0]["source_range_seconds"] == [0.0, 2.0]


def test_scene_fidelity_allows_non_overlapping_source_reuse():
    registry = load_registry()

    report = check_plan(
        {
            "version": "1.0",
            "render_runtime": "ffmpeg",
            "cuts": [
                {
                    "id": "cut-1",
                    "source": "projects/example/assets/video/clip.mp4",
                    "in_seconds": 0.0,
                    "out_seconds": 2.0,
                    "source_in_seconds": 0.0,
                    "maps_to_beat": "hook",
                },
                {
                    "id": "cut-2",
                    "source": "projects/example/assets/video/clip.mp4",
                    "in_seconds": 2.0,
                    "out_seconds": 4.0,
                    "source_in_seconds": 2.0,
                    "maps_to_beat": "build",
                },
            ],
        },
        registry,
    )

    assert report["ok"], report["issues"]


def test_scene_fidelity_allows_reused_still_image_in_remotion_cuts():
    registry = load_registry()

    report = check_plan(
        {
            "version": "1.0",
            "render_runtime": "remotion",
            "cuts": [
                {
                    "id": "cut-1",
                    "source": "projects/example/assets/images/packshot.png",
                    "in_seconds": 0.0,
                    "out_seconds": 2.0,
                    "maps_to_beat": "hook",
                },
                {
                    "id": "cut-2",
                    "source": "projects/example/assets/images/packshot.png",
                    "in_seconds": 2.0,
                    "out_seconds": 4.0,
                    "maps_to_beat": "cta",
                },
            ],
        },
        registry,
    )

    assert report["ok"], report["issues"]


def test_video_compose_precompose_blocks_overlapping_source_reuse():
    composer = VideoCompose()

    result = composer._pre_compose_validation(
        {
            "version": "1.0",
            "render_runtime": "ffmpeg",
            "renderer_family": "cinematic-trailer",
            "cuts": [
                {
                    "id": "cut-1",
                    "source": "projects/example/assets/video/clip.mp4",
                    "in_seconds": 0.0,
                    "out_seconds": 2.0,
                    "source_in_seconds": 0.0,
                    "maps_to_beat": "hook",
                },
                {
                    "id": "cut-2",
                    "source": "projects/example/assets/video/clip.mp4",
                    "in_seconds": 2.0,
                    "out_seconds": 4.0,
                    "source_in_seconds": 1.0,
                    "maps_to_beat": "build",
                },
            ],
        },
        [
            {
                "id": "cut-1",
                "source": "projects/example/assets/video/clip.mp4",
                "in_seconds": 0.0,
                "out_seconds": 2.0,
                "source_in_seconds": 0.0,
                "maps_to_beat": "hook",
            },
            {
                "id": "cut-2",
                "source": "projects/example/assets/video/clip.mp4",
                "in_seconds": 2.0,
                "out_seconds": 4.0,
                "source_in_seconds": 1.0,
                "maps_to_beat": "build",
            },
        ],
    )

    assert result is not None
    assert not result.success
    assert "overlapping cut 'cut-1'" in result.error


def test_video_compose_accepts_subtitle_style_string_from_schema():
    style = VideoCompose._resolve_subtitle_style(
        explicit_style=None,
        edit_decisions={
            "version": "1.0",
            "render_runtime": "ffmpeg",
            "subtitles": {
                "enabled": True,
                "style": "sentence",
                "font": "DejaVu Sans",
                "font_size": 24,
            },
            "cuts": [],
        },
        playbook=None,
    )

    assert style["font"] == "DejaVu Sans"
    assert style["font_size"] == 24


def test_video_compose_derives_theme_from_schema_v2_playbook_fields():
    theme = VideoCompose._build_theme_from_playbook("anime-ghibli", None)

    assert theme is not None
    assert theme["headingFont"] == "Noto Serif JP"
    assert theme["bodyFont"] == "Noto Sans"
    assert theme["mutedTextColor"] == "#8B9A7E"
    assert theme["chartColors"][:3] == ["#A8E6CF", "#FFB347", "#FF6B9D"]
    assert theme["transitionDuration"] == 1.0


def test_video_compose_derives_theme_from_ad_brand_playbook():
    theme = VideoCompose._build_theme_from_playbook("ad-brand", None)

    assert theme is not None
    assert theme["primaryColor"] == "#1A1A2E"
    assert theme["accentColor"] == "#E94560"
    assert theme["headingFont"] == "Inter"
    assert theme["bodyFont"] == "Inter"
    assert theme["chartColors"][:3] == ["#E94560", "#0F3460", "#533483"]
    assert theme["transitionDuration"] == 0.2


def test_video_compose_derives_subtitle_font_from_playbook_body_font():
    style = VideoCompose._resolve_subtitle_style(
        explicit_style=None,
        edit_decisions=None,
        playbook=load_playbook("anime-ghibli"),
    )

    assert style["font"] == "Noto Sans"
    assert style["primary_color"] == "#F5F0E8"


def test_video_compose_converts_hex_subtitle_colors_to_ass_override():
    force_style = VideoCompose._build_subtitle_style(
        {
            "font": "DejaVu Sans",
            "font_size": 24,
            "primary_color": "#FFFFFF",
            "outline_color": "#000000",
        }
    )

    assert "PrimaryColour=&H00FFFFFF" in force_style
    assert "OutlineColour=&H00000000" in force_style
    assert "#FFFFFF" not in force_style


def test_video_compose_converts_css_alpha_hex_subtitle_color_to_ass_transparency():
    force_style = VideoCompose._build_subtitle_style(
        {
            "primary_color": "#00000000",
            "outline_color": "#33669980",
            "back_color": "#FFFFFF40",
        }
    )

    assert "PrimaryColour=&HFF000000" in force_style
    assert "OutlineColour=&H7F996633" in force_style
    assert "BackColour=&HBFFFFFFF" in force_style
