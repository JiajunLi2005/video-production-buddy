from __future__ import annotations

import ast
from pathlib import Path, PureWindowsPath
from typing import Any, Callable

import pytest

from tools.base_tool import ToolResult
from tools.output_paths import (
    portable_output_path,
    require_explicit_output_path,
    require_explicit_project_artifact_destination,
    require_explicit_project_corpus_destination,
    require_explicit_project_media_destination,
    require_explicit_project_sidecar_destination,
    require_explicit_project_sidecar_path,
    require_explicit_project_source_media_destination,
    require_optional_project_artifact_destination,
    require_optional_project_media_directory_destination,
    require_optional_project_media_destination,
    require_optional_project_sidecar_path,
)
from tools.tool_registry import ToolRegistry

PathHelper = Callable[..., tuple[Path | None, ToolResult | None]]

EXPLICIT_OUTPUT_HELPERS = {
    "require_explicit_output_path",
    "require_explicit_project_artifact_destination",
    "require_explicit_project_media_destination",
    "require_explicit_project_media_directory_destination",
    "require_explicit_project_sidecar_destination",
    "require_explicit_project_sidecar_path",
    "require_explicit_project_source_media_destination",
    "require_explicit_project_corpus_destination",
}


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            PureWindowsPath("projects", "demo", "renders", "final.mp4"),
            "projects/demo/renders/final.mp4",
        ),
        (
            PureWindowsPath("C:/workspace/projects/demo/artifacts/report.json"),
            "C:/workspace/projects/demo/artifacts/report.json",
        ),
    ],
)
def test_portable_output_path_normalizes_windows_separators(
    path: PureWindowsPath,
    expected: str,
) -> None:
    assert portable_output_path(path) == expected


@pytest.mark.parametrize(
    ("helper", "field_name"),
    [
        (require_explicit_output_path, "output_path"),
        (require_optional_project_sidecar_path, "sidecar_path"),
        (require_optional_project_artifact_destination, "artifact_dir"),
        (require_explicit_project_artifact_destination, "artifact_dir"),
        (require_optional_project_media_destination, "media_dir"),
        (require_optional_project_media_directory_destination, "media_dir"),
        (require_explicit_project_sidecar_destination, "sidecar_dir"),
        (require_explicit_project_media_destination, "media_dir"),
        (require_explicit_project_source_media_destination, "source_dir"),
        (require_explicit_project_corpus_destination, "corpus_dir"),
        (require_explicit_project_sidecar_path, "sidecar_path"),
    ],
)
@pytest.mark.parametrize("raw_path", [[], {}, 123])
def test_project_path_helpers_reject_non_string_values(
    helper: PathHelper,
    field_name: str,
    raw_path: object,
) -> None:
    path, error = _call_path_helper(helper, field_name, raw_path)

    assert path is None
    assert error is not None
    assert f"{field_name} for demo artifact must be a string path" in (
        error.error or ""
    )


def test_generated_media_output_path_accepts_pathlike_value() -> None:
    path, error = require_explicit_output_path(
        {"output_path": Path("projects/demo/assets/audio/voice.mp3")},
        "demo_tool",
        artifact_label="demo artifact",
    )

    assert error is None
    assert path == Path("projects/demo/assets/audio/voice.mp3")


@pytest.mark.parametrize(
    "raw_path",
    [
        " projects/demo/assets/audio/voice.mp3",
        "projects/demo/assets/audio/voice.mp3 ",
    ],
)
def test_generated_media_output_path_rejects_padded_project_paths(
    raw_path: str,
) -> None:
    path, error = require_explicit_output_path(
        {"output_path": raw_path},
        "demo_tool",
        artifact_label="demo artifact",
    )

    assert path is None
    assert error is not None
    assert "projects/<project-name>/" in (error.error or "")


@pytest.mark.parametrize(
    ("helper", "field_name"),
    [
        (require_optional_project_sidecar_path, "sidecar_path"),
        (require_explicit_project_sidecar_path, "sidecar_path"),
    ],
)
def test_project_sidecar_path_helpers_require_file_paths(
    helper: PathHelper,
    field_name: str,
) -> None:
    path, error = _call_path_helper(
        helper,
        field_name,
        "projects/demo/artifacts/transcript",
    )

    assert path is None
    assert error is not None
    assert "must be a file path" in (error.error or "")


def test_project_sidecar_destination_accepts_directory_path() -> None:
    path, error = require_explicit_project_sidecar_destination(
        {"sidecar_dir": "projects/demo/artifacts/transcripts"},
        "sidecar_dir",
        "demo_tool",
        artifact_label="demo artifact",
    )

    assert error is None
    assert path == Path("projects/demo/artifacts/transcripts")


def test_project_sidecar_destination_rejects_file_paths() -> None:
    path, error = require_explicit_project_sidecar_destination(
        {"sidecar_dir": "projects/demo/artifacts/transcript.json"},
        "sidecar_dir",
        "demo_tool",
        artifact_label="demo artifact",
    )

    assert path is None
    assert error is not None
    assert "sidecar_dir for demo artifact must be a directory path" in (
        error.error or ""
    )
    assert "projects/<project-name>/artifacts/" in (error.error or "")


@pytest.mark.parametrize(
    ("helper", "field_name", "raw_path"),
    [
        (
            require_optional_project_artifact_destination,
            "artifact_path",
            "projects/demo/artifacts/character_design.json",
        ),
        (
            require_explicit_project_artifact_destination,
            "artifact_path",
            "projects/demo/artifacts/character_design.json",
        ),
        (
            require_optional_project_media_destination,
            "media_path",
            "projects/demo/renders/preview.mp4",
        ),
    ],
)
def test_flexible_destination_helpers_accept_file_paths(
    helper: PathHelper,
    field_name: str,
    raw_path: str,
) -> None:
    path, error = _call_path_helper(helper, field_name, raw_path)

    assert error is None
    assert path == Path(raw_path)


def test_optional_media_directory_destination_rejects_file_paths() -> None:
    path, error = require_optional_project_media_directory_destination(
        {"media_dir": "projects/demo/renders/hyperframes.json"},
        "media_dir",
        "demo_tool",
        artifact_label="demo workspace",
    )

    assert path is None
    assert error is not None
    assert "media_dir for demo workspace must be a directory path" in (
        error.error or ""
    )


def test_explicit_project_artifact_destination_requires_path() -> None:
    path, error = require_explicit_project_artifact_destination(
        {},
        "artifact_dir",
        "demo_tool",
        artifact_label="demo artifact",
    )

    assert path is None
    assert error is not None
    assert "artifact_dir is required for demo artifact" in (error.error or "")
    assert "projects/<project-name>/artifacts/" in (error.error or "")


def test_explicit_output_helpers_are_reflected_in_tool_input_schemas() -> None:
    failures: list[str] = []
    tools_root = Path(__file__).resolve().parents[2] / "tools"

    for path in sorted(tools_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            explicit_fields = _explicit_output_fields(cls)
            if not explicit_fields:
                continue
            input_schema = _class_assignment(cls, "input_schema")
            required_mentions = (
                _schema_required_mentions(input_schema) if input_schema else set()
            )
            missing = sorted(explicit_fields - required_mentions)
            if missing:
                failures.append(
                    f"{path.relative_to(tools_root.parent)}:{cls.name} "
                    f"does not mark {missing} required in input_schema"
                )

    assert failures == []


def test_explicit_output_helpers_are_reflected_in_idempotency_keys() -> None:
    failures: list[str] = []
    tools_root = Path(__file__).resolve().parents[2] / "tools"

    for path in sorted(tools_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            explicit_fields = _explicit_output_fields(cls)
            if not explicit_fields:
                continue
            idempotency_fields = _constant_string_list(
                _class_assignment(cls, "idempotency_key_fields")
            )
            missing = sorted(explicit_fields - idempotency_fields)
            if missing:
                failures.append(
                    f"{path.relative_to(tools_root.parent)}:{cls.name} "
                    f"does not include {missing} in idempotency_key_fields"
                )

    assert failures == []


def test_explicit_output_path_helpers_are_reflected_in_output_schemas() -> None:
    failures: list[str] = []
    tools_root = Path(__file__).resolve().parents[2] / "tools"

    for path in sorted(tools_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            if "output_path" not in _explicit_output_fields(cls):
                continue
            output_schema = _class_assignment(cls, "output_schema")
            if not output_schema or "output_path" not in _schema_required_mentions(
                output_schema
            ):
                failures.append(
                    f"{path.relative_to(tools_root.parent)}:{cls.name} "
                    "uses an explicit output_path helper but output_schema does "
                    "not require output_path"
                )

    assert failures == []


def test_explicit_output_helpers_are_reflected_in_runtime_output_schema_properties() -> None:
    failures: list[str] = []
    tools_root = Path(__file__).resolve().parents[2] / "tools"
    registry = ToolRegistry()
    registry.discover()
    tools_by_class = {
        (tool.__class__.__module__, tool.__class__.__name__): tool
        for tool in registry._tools.values()
    }

    for path in sorted(tools_root.rglob("*.py")):
        if path.name.startswith("__"):
            continue
        module_name = ".".join(path.with_suffix("").parts)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
            explicit_fields = _explicit_output_fields(cls)
            if not explicit_fields:
                continue
            tool = tools_by_class.get((module_name, cls.name))
            if tool is None:
                continue
            output_properties = set(
                ((tool.output_schema or {}).get("properties") or {}).keys()
            )
            missing = sorted(explicit_fields - output_properties)
            if missing:
                failures.append(
                    f"{path.relative_to(tools_root.parent)}:{cls.name} "
                    f"uses an explicit destination helper but output_schema "
                    f"does not declare {missing}"
                )

    assert failures == []


@pytest.mark.parametrize(
    ("helper", "field_name", "raw_path", "expected_hint"),
    [
        (
            require_explicit_project_source_media_destination,
            "source_dir",
            "projects/demo/assets/video/raw.mp4",
            "projects/<project-name>/reference_assets/",
        ),
        (
            require_explicit_project_corpus_destination,
            "corpus_dir",
            "projects/demo/corpus/index.jsonl",
            "projects/<project-name>/corpus/",
        ),
    ],
)
def test_project_directory_destination_helpers_reject_file_paths(
    helper: PathHelper,
    field_name: str,
    raw_path: str,
    expected_hint: str,
) -> None:
    path, error = _call_path_helper(helper, field_name, raw_path)

    assert path is None
    assert error is not None
    assert f"{field_name} for demo artifact must be a directory path" in (
        error.error or ""
    )
    assert expected_hint in (error.error or "")


def _call_path_helper(
    helper: PathHelper,
    field_name: str,
    raw_path: Any,
) -> tuple[Path | None, ToolResult | None]:
    if helper is require_explicit_output_path:
        return helper(
            {"output_path": raw_path},
            "demo_tool",
            artifact_label="demo artifact",
        )
    return helper(
        {field_name: raw_path},
        field_name,
        "demo_tool",
        artifact_label="demo artifact",
    )


def _class_assignment(cls: ast.ClassDef, name: str) -> ast.AST | None:
    for stmt in cls.body:
        value: ast.AST | None = None
        targets: list[ast.AST] = []
        if isinstance(stmt, ast.Assign):
            value = stmt.value
            targets = list(stmt.targets)
        elif isinstance(stmt, ast.AnnAssign):
            value = stmt.value
            targets = [stmt.target]
        if value is None:
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return value
    return None


def _explicit_output_fields(cls: ast.ClassDef) -> set[str]:
    fields: set[str] = set()
    for node in ast.walk(cls):
        if not isinstance(node, ast.Call):
            continue
        function_name = _call_name(node)
        if function_name not in EXPLICIT_OUTPUT_HELPERS:
            continue
        if function_name == "require_explicit_output_path":
            fields.add("output_path")
        elif len(node.args) >= 2:
            field_name = _constant_string(node.args[1])
            if field_name:
                fields.add(field_name)
    return fields


def _schema_required_mentions(schema: ast.AST) -> set[str]:
    required: set[str] = set()
    for node in ast.walk(schema):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if _constant_string(key) != "required":
                continue
            if isinstance(value, (ast.List, ast.Tuple)):
                required.update(
                    field
                    for item in value.elts
                    if (field := _constant_string(item))
                )
    return required


def _constant_string_list(node: ast.AST | None) -> set[str]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return set()
    return {
        value
        for item in node.elts
        if (value := _constant_string(item))
    }


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _constant_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None
