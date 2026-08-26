"""Output path validation helpers for tools that write project artifacts."""

from __future__ import annotations

from os import PathLike, fspath
from pathlib import Path, PurePath
from typing import Any

from tools.base_tool import ToolResult

_GENERATED_MEDIA_ROOTS = {"assets", "renders"}
_PROJECT_ARTIFACT_ROOTS = {"artifacts"}
_PROJECT_SIDECAR_ROOTS = {"artifacts", "assets", "renders"}
_SOURCE_MEDIA_ROOTS = {"assets", "reference_assets"}
_PROJECT_CORPUS_ROOTS = {"corpus"}


def portable_output_path(path: PurePath) -> str:
    """Serialize a filesystem path for persisted tool output metadata.

    Tools should keep using native ``Path`` objects for filesystem and subprocess
    operations. At JSON, artifact, and ``ToolResult`` boundaries, POSIX-style
    separators keep project-relative references stable across operating systems.
    """
    return path.as_posix()


def require_explicit_output_path(
    inputs: dict[str, Any],
    tool_name: str,
    *,
    artifact_label: str = "generated media",
) -> tuple[Path | None, ToolResult | None]:
    """Require callers to choose where generated artifacts are written.

    Generated media tools should not silently write files like ``tts_output.mp3``
    into the repository root. Production callers are expected to pass a
    project-scoped path such as ``projects/<project-name>/assets/audio/...``.
    """
    raw_output_path = inputs.get("output_path")
    if _is_missing_path_input(raw_output_path):
        return None, ToolResult(
            success=False,
            error=(
                f"{tool_name}: output_path is required for {artifact_label}; "
                "write generated media under "
                "projects/<project-name>/assets/... or projects/<project-name>/renders/..."
            ),
        )
    output_path, path_error = _coerce_path_input(
        raw_output_path,
        field_name="output_path",
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots_text="projects/<project-name>/assets/... or projects/<project-name>/renders/...",
    )
    if path_error:
        return None, path_error
    assert output_path is not None
    if not output_path.suffix:
        return None, ToolResult(
            success=False,
            error=(
                f"{tool_name}: output_path for {artifact_label} must be a file path "
                "under projects/<project-name>/assets/... or projects/<project-name>/renders/..."
            ),
        )
    error = _validate_project_path(
        output_path,
        field_name="output_path",
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots=_GENERATED_MEDIA_ROOTS,
        allowed_roots_text="projects/<project-name>/assets/... or projects/<project-name>/renders/...",
    )
    if error:
        return None, error
    return output_path, None


def require_optional_project_sidecar_path(
    inputs: dict[str, Any],
    field_name: str,
    tool_name: str,
    *,
    artifact_label: str = "sidecar artifact",
) -> tuple[Path | None, ToolResult | None]:
    raw_path = inputs.get(field_name)
    if _is_missing_path_input(raw_path):
        return None, None
    path, path_error = _coerce_path_input(
        raw_path,
        field_name=field_name,
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots_text=(
            "projects/<project-name>/artifacts/..., "
            "projects/<project-name>/assets/..., or "
            "projects/<project-name>/renders/..."
        ),
    )
    if path_error:
        return None, path_error
    assert path is not None
    file_error = _require_file_path(
        path,
        field_name=field_name,
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots_text=(
            "projects/<project-name>/artifacts/..., "
            "projects/<project-name>/assets/..., or "
            "projects/<project-name>/renders/..."
        ),
    )
    if file_error:
        return None, file_error
    error = _validate_project_path(
        path,
        field_name=field_name,
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots=_PROJECT_SIDECAR_ROOTS,
        allowed_roots_text=(
            "projects/<project-name>/artifacts/..., "
            "projects/<project-name>/assets/..., or "
            "projects/<project-name>/renders/..."
        ),
    )
    if error:
        return None, error
    return path, None


def require_optional_project_artifact_destination(
    inputs: dict[str, Any],
    field_name: str,
    tool_name: str,
    *,
    artifact_label: str = "canonical artifact",
) -> tuple[Path | None, ToolResult | None]:
    raw_path = inputs.get(field_name)
    if _is_missing_path_input(raw_path):
        return None, None
    path, path_error = _coerce_path_input(
        raw_path,
        field_name=field_name,
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots_text="projects/<project-name>/artifacts/...",
    )
    if path_error:
        return None, path_error
    assert path is not None
    validation_path = path if path.suffix else path / "__output__"
    error = _validate_project_path(
        validation_path,
        field_name=field_name,
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots=_PROJECT_ARTIFACT_ROOTS,
        allowed_roots_text="projects/<project-name>/artifacts/...",
    )
    if error:
        return None, error
    return path, None


def require_explicit_project_artifact_destination(
    inputs: dict[str, Any],
    field_name: str,
    tool_name: str,
    *,
    artifact_label: str = "canonical artifact",
) -> tuple[Path | None, ToolResult | None]:
    raw_path = inputs.get(field_name)
    if _is_missing_path_input(raw_path):
        return None, ToolResult(
            success=False,
            error=(
                f"{tool_name}: {field_name} is required for {artifact_label}; "
                "write canonical artifacts under projects/<project-name>/artifacts/..."
            ),
        )
    return require_optional_project_artifact_destination(
        inputs,
        field_name,
        tool_name,
        artifact_label=artifact_label,
    )


def require_optional_project_media_destination(
    inputs: dict[str, Any],
    field_name: str,
    tool_name: str,
    *,
    artifact_label: str = "generated media",
) -> tuple[Path | None, ToolResult | None]:
    raw_path = inputs.get(field_name)
    if _is_missing_path_input(raw_path):
        return None, None
    path, path_error = _coerce_path_input(
        raw_path,
        field_name=field_name,
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots_text="projects/<project-name>/assets/... or projects/<project-name>/renders/...",
    )
    if path_error:
        return None, path_error
    assert path is not None
    validation_path = path if path.suffix else path / "__output__"
    error = _validate_project_path(
        validation_path,
        field_name=field_name,
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots=_GENERATED_MEDIA_ROOTS,
        allowed_roots_text="projects/<project-name>/assets/... or projects/<project-name>/renders/...",
    )
    if error:
        return None, error
    return path, None


def require_optional_project_media_directory_destination(
    inputs: dict[str, Any],
    field_name: str,
    tool_name: str,
    *,
    artifact_label: str = "generated media",
) -> tuple[Path | None, ToolResult | None]:
    path, path_error = require_optional_project_media_destination(
        inputs,
        field_name,
        tool_name,
        artifact_label=artifact_label,
    )
    if path_error:
        return None, path_error
    if path is None:
        return None, None
    directory_error = _require_directory_path(
        path,
        field_name=field_name,
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots_text="projects/<project-name>/assets/... or projects/<project-name>/renders/...",
    )
    if directory_error:
        return None, directory_error
    return path, None


def require_explicit_project_sidecar_destination(
    inputs: dict[str, Any],
    field_name: str,
    tool_name: str,
    *,
    artifact_label: str = "sidecar artifact",
) -> tuple[Path | None, ToolResult | None]:
    raw_path = inputs.get(field_name)
    if _is_missing_path_input(raw_path):
        return None, ToolResult(
            success=False,
            error=(
                f"{tool_name}: {field_name} is required for {artifact_label}; "
                "write sidecar artifacts under "
                "projects/<project-name>/artifacts/..., "
                "projects/<project-name>/assets/..., or "
                "projects/<project-name>/renders/..."
            ),
        )
    path, path_error = _coerce_path_input(
        raw_path,
        field_name=field_name,
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots_text=(
            "projects/<project-name>/artifacts/..., "
            "projects/<project-name>/assets/..., or "
            "projects/<project-name>/renders/..."
        ),
    )
    if path_error:
        return None, path_error
    assert path is not None
    directory_error = _require_directory_path(
        path,
        field_name=field_name,
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots_text=(
            "projects/<project-name>/artifacts/..., "
            "projects/<project-name>/assets/..., or "
            "projects/<project-name>/renders/..."
        ),
    )
    if directory_error:
        return None, directory_error
    validation_path = path if path.suffix else path / "__output__"
    error = _validate_project_path(
        validation_path,
        field_name=field_name,
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots=_PROJECT_SIDECAR_ROOTS,
        allowed_roots_text=(
            "projects/<project-name>/artifacts/..., "
            "projects/<project-name>/assets/..., or "
            "projects/<project-name>/renders/..."
        ),
    )
    if error:
        return None, error
    return path, None


def require_explicit_project_media_destination(
    inputs: dict[str, Any],
    field_name: str,
    tool_name: str,
    *,
    artifact_label: str = "generated media",
) -> tuple[Path | None, ToolResult | None]:
    raw_path = inputs.get(field_name)
    if _is_missing_path_input(raw_path):
        return None, ToolResult(
            success=False,
            error=(
                f"{tool_name}: {field_name} is required for {artifact_label}; "
                "write generated media under "
                "projects/<project-name>/assets/... or projects/<project-name>/renders/..."
            ),
        )
    return require_optional_project_media_destination(
        inputs,
        field_name,
        tool_name,
        artifact_label=artifact_label,
    )


def require_explicit_project_media_directory_destination(
    inputs: dict[str, Any],
    field_name: str,
    tool_name: str,
    *,
    artifact_label: str = "generated media",
) -> tuple[Path | None, ToolResult | None]:
    path, path_error = require_explicit_project_media_destination(
        inputs,
        field_name,
        tool_name,
        artifact_label=artifact_label,
    )
    if path_error:
        return None, path_error
    assert path is not None
    directory_error = _require_directory_path(
        path,
        field_name=field_name,
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots_text="projects/<project-name>/assets/... or projects/<project-name>/renders/...",
    )
    if directory_error:
        return None, directory_error
    return path, None


def require_explicit_project_source_media_destination(
    inputs: dict[str, Any],
    field_name: str,
    tool_name: str,
    *,
    artifact_label: str = "source media",
) -> tuple[Path | None, ToolResult | None]:
    raw_path = inputs.get(field_name)
    if _is_missing_path_input(raw_path):
        return None, ToolResult(
            success=False,
            error=(
                f"{tool_name}: {field_name} is required for {artifact_label}; "
                "write source or reference media under "
                "projects/<project-name>/reference_assets/... or "
                "projects/<project-name>/assets/..."
            ),
        )
    path, path_error = _coerce_path_input(
        raw_path,
        field_name=field_name,
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots_text=(
            "projects/<project-name>/reference_assets/... or "
            "projects/<project-name>/assets/..."
        ),
    )
    if path_error:
        return None, path_error
    assert path is not None
    directory_error = _require_directory_path(
        path,
        field_name=field_name,
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots_text=(
            "projects/<project-name>/reference_assets/... or "
            "projects/<project-name>/assets/..."
        ),
    )
    if directory_error:
        return None, directory_error
    validation_path = path if path.suffix else path / "__output__"
    error = _validate_project_path(
        validation_path,
        field_name=field_name,
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots=_SOURCE_MEDIA_ROOTS,
        allowed_roots_text=(
            "projects/<project-name>/reference_assets/... or "
            "projects/<project-name>/assets/..."
        ),
    )
    if error:
        return None, error
    return path, None


def require_explicit_project_corpus_destination(
    inputs: dict[str, Any],
    field_name: str,
    tool_name: str,
    *,
    artifact_label: str = "clip corpus",
) -> tuple[Path | None, ToolResult | None]:
    raw_path = inputs.get(field_name)
    if _is_missing_path_input(raw_path):
        return None, ToolResult(
            success=False,
            error=(
                f"{tool_name}: {field_name} is required for {artifact_label}; "
                "write corpus data under projects/<project-name>/corpus/..."
            ),
        )
    path, path_error = _coerce_path_input(
        raw_path,
        field_name=field_name,
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots_text="projects/<project-name>/corpus/...",
    )
    if path_error:
        return None, path_error
    assert path is not None
    directory_error = _require_directory_path(
        path,
        field_name=field_name,
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots_text="projects/<project-name>/corpus/...",
    )
    if directory_error:
        return None, directory_error
    validation_path = path if path.suffix else path / "__output__"
    error = _validate_project_path(
        validation_path,
        field_name=field_name,
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots=_PROJECT_CORPUS_ROOTS,
        allowed_roots_text="projects/<project-name>/corpus/...",
    )
    if error:
        return None, error
    return path, None


def require_explicit_project_sidecar_path(
    inputs: dict[str, Any],
    field_name: str,
    tool_name: str,
    *,
    artifact_label: str = "sidecar artifact",
) -> tuple[Path | None, ToolResult | None]:
    raw_path = inputs.get(field_name)
    if _is_missing_path_input(raw_path):
        return None, ToolResult(
            success=False,
            error=(
                f"{tool_name}: {field_name} is required for {artifact_label}; "
                "write sidecar artifacts under "
                "projects/<project-name>/artifacts/..., "
                "projects/<project-name>/assets/..., or "
                "projects/<project-name>/renders/..."
            ),
        )
    path, path_error = _coerce_path_input(
        raw_path,
        field_name=field_name,
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots_text=(
            "projects/<project-name>/artifacts/..., "
            "projects/<project-name>/assets/..., or "
            "projects/<project-name>/renders/..."
        ),
    )
    if path_error:
        return None, path_error
    assert path is not None
    file_error = _require_file_path(
        path,
        field_name=field_name,
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots_text=(
            "projects/<project-name>/artifacts/..., "
            "projects/<project-name>/assets/..., or "
            "projects/<project-name>/renders/..."
        ),
    )
    if file_error:
        return None, file_error
    error = _validate_project_path(
        path,
        field_name=field_name,
        tool_name=tool_name,
        artifact_label=artifact_label,
        allowed_roots=_PROJECT_SIDECAR_ROOTS,
        allowed_roots_text=(
            "projects/<project-name>/artifacts/..., "
            "projects/<project-name>/assets/..., or "
            "projects/<project-name>/renders/..."
        ),
    )
    if error:
        return None, error
    return path, None


def _is_missing_path_input(raw_path: Any) -> bool:
    if raw_path is None:
        return True
    try:
        path_value = fspath(raw_path) if isinstance(raw_path, PathLike) else raw_path
    except TypeError:
        return False
    return isinstance(path_value, str) and not path_value.strip()


def _coerce_path_input(
    raw_path: Any,
    *,
    field_name: str,
    tool_name: str,
    artifact_label: str,
    allowed_roots_text: str,
) -> tuple[Path | None, ToolResult | None]:
    try:
        path_value = fspath(raw_path) if isinstance(raw_path, PathLike) else raw_path
    except TypeError:
        path_value = raw_path
    if not isinstance(path_value, str):
        return None, ToolResult(
            success=False,
            error=(
                f"{tool_name}: {field_name} for {artifact_label} must be a string path "
                f"under {allowed_roots_text}"
            ),
        )
    path_text = path_value.strip()
    if not path_text:
        return None, None
    if path_text != path_value:
        return None, ToolResult(
            success=False,
            error=(
                f"{tool_name}: {field_name} for {artifact_label} must stay under "
                f"{allowed_roots_text}"
            ),
        )
    return Path(path_value), None


def _require_file_path(
    path: Path,
    *,
    field_name: str,
    tool_name: str,
    artifact_label: str,
    allowed_roots_text: str,
) -> ToolResult | None:
    if path.suffix:
        return None
    return ToolResult(
        success=False,
        error=(
            f"{tool_name}: {field_name} for {artifact_label} must be a file path "
            f"under {allowed_roots_text}"
        ),
    )


def _require_directory_path(
    path: Path,
    *,
    field_name: str,
    tool_name: str,
    artifact_label: str,
    allowed_roots_text: str,
) -> ToolResult | None:
    if not path.suffix:
        return None
    return ToolResult(
        success=False,
        error=(
            f"{tool_name}: {field_name} for {artifact_label} must be a directory path "
            f"under {allowed_roots_text}"
        ),
    )


def _validate_project_path(
    output_path: Path,
    *,
    field_name: str,
    tool_name: str,
    artifact_label: str,
    allowed_roots: set[str],
    allowed_roots_text: str,
) -> ToolResult | None:
    if ".." in output_path.parts:
        return ToolResult(
            success=False,
            error=(
                f"{tool_name}: {field_name} for {artifact_label} must stay under "
                f"{allowed_roots_text}"
            ),
        )
    workspace_relative = _workspace_relative_output_path(output_path)
    parts = workspace_relative.parts
    if (
        len(parts) < 4
        or parts[0] != "projects"
        or not parts[1]
        or parts[2] not in allowed_roots
    ):
        return ToolResult(
            success=False,
            error=(
                f"{tool_name}: {field_name} for {artifact_label} must stay under "
                f"{allowed_roots_text}"
            ),
        )
    return None


def _workspace_relative_output_path(path: Path) -> Path:
    workspace = Path.cwd().resolve()
    if path.is_absolute():
        try:
            return path.resolve(strict=False).relative_to(workspace)
        except ValueError:
            return path
    try:
        return (workspace / path).resolve(strict=False).relative_to(workspace)
    except ValueError:
        return path
