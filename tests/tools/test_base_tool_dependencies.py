from __future__ import annotations

import ast
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from fractions import Fraction
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import UUID

import pytest
from jsonschema.validators import validator_for

from tools.base_tool import (
    BaseTool,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)
from tools.status_utils import safe_tool_info
from tools.tool_registry import ToolRegistry


class BinaryDependencyTool(BaseTool):
    name = "binary_dependency_tool"
    dependencies = ["binary:definitely-not-a-real-video-production-buddy-command"]

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class UnknownDependencyTool(BaseTool):
    name = "unknown_dependency_tool"
    dependencies = ["unknown:definitely-not-real"]

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class EnvAnyDependencyTool(BaseTool):
    name = "env_any_dependency_tool"
    dependencies = [
        "env_any:VIDEO_PRODUCTION_BUDDY_PRIMARY_TEST_KEY,VIDEO_PRODUCTION_BUDDY_SECONDARY_TEST_KEY"
    ]

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class EnvAnySetupTool(BaseTool):
    name = "env_any_setup_tool"
    capability = "video_generation"
    provider = "env_any_provider"
    runtime = ToolRuntime.API
    dependencies = [
        "env_any:VIDEO_PRODUCTION_BUDDY_PRIMARY_TEST_KEY,VIDEO_PRODUCTION_BUDDY_SECONDARY_TEST_KEY"
    ]
    install_instructions = (
        "Set VIDEO_PRODUCTION_BUDDY_PRIMARY_TEST_KEY or VIDEO_PRODUCTION_BUDDY_SECONDARY_TEST_KEY "
        "to your API key."
    )

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class PathIdempotencyTool(BaseTool):
    name = "path_idempotency_tool"
    input_schema = {
        "type": "object",
        "properties": {"output_path": {"type": "string"}},
    }
    progress_schema = {
        "type": "object",
        "properties": {"percent": {"type": "number"}},
    }
    retry_policy = RetryPolicy(
        max_retries=2,
        backoff_seconds=3.0,
        retryable_errors=["timeout"],
    )
    idempotency_key_fields = ["output_path"]

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class ComplexIdempotencyTool(BaseTool):
    name = "complex_idempotency_tool"
    input_schema = {
        "type": "object",
        "properties": {"payload": {"type": "object"}},
    }
    idempotency_key_fields = ["payload"]

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class DefaultMutableContractToolA(BaseTool):
    name = "default_mutable_contract_tool_a"

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class DefaultMutableContractToolB(BaseTool):
    name = "default_mutable_contract_tool_b"

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class DuplicateNameToolA(BaseTool):
    name = "duplicate_name_tool"

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class DuplicateNameToolB(BaseTool):
    name = "duplicate_name_tool"

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class LocalResultEnum(Enum):
    READY = "ready"


class CustomResultPayload:
    def __str__(self) -> str:
        return "custom-result-payload"


class NonFiniteTelemetryTool(BaseTool):
    name = "non_finite_telemetry_tool"
    quality_score = math.nan
    historical_success_rate = math.inf
    latency_p50_seconds = -math.inf

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class NonFiniteDryRunTool(BaseTool):
    name = "non_finite_dry_run_tool"

    def estimate_cost(self, inputs: dict) -> float:
        return math.inf

    def estimate_runtime(self, inputs: dict) -> float:
        return math.nan

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class BrokenEstimateDryRunTool(BaseTool):
    name = "broken_estimate_dry_run_tool"

    def estimate_cost(self, inputs: dict) -> float:
        raise RuntimeError("cost estimator unavailable")

    def estimate_runtime(self, inputs: dict) -> float:
        raise ValueError("runtime estimator unavailable")

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class BrokenStatusTool(BaseTool):
    name = "broken_status_tool"
    capability = "video_generation"
    provider = "broken_provider"
    best_for = ["testing broken status isolation"]

    def get_status(self) -> ToolStatus:
        raise RuntimeError("status backend unavailable")

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class StringStatusTool(BaseTool):
    name = "string_status_tool"
    capability = "video_generation"
    provider = "string_provider"

    def get_status(self):
        return "available"

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class NonJsonBrokenStatusTool(BaseTool):
    name = "non_json_broken_status_tool"
    capability = "video_generation"
    provider = "broken_provider"
    best_for = [PureWindowsPath(r"docs\provider.md"), Decimal("1.25")]
    dependencies = [PureWindowsPath(".env")]
    provider_matrix = {"fallback": {"cost": Decimal("0.10"), "score": math.nan}}
    quality_score = math.nan
    user_visible_verification = [b"listen"]

    def get_status(self) -> ToolStatus:
        raise RuntimeError("status backend unavailable")

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class QueryableMetadataTool(BaseTool):
    name = "queryable_metadata_tool"
    tier = ToolTier.VOICE
    stability = ToolStability.PRODUCTION
    capability = "query_capability"
    provider = "query_provider"
    capabilities = ["query_alias"]
    resource_profile = ResourceProfile(vram_mb=256, network_required=True)

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class BrokenRegistryQueryMetadataTool(BaseTool):
    name = "broken_registry_query_metadata_tool"

    @property
    def tier(self) -> ToolTier:
        raise RuntimeError("tier metadata unavailable")

    @property
    def stability(self) -> ToolStability:
        raise RuntimeError("stability metadata unavailable")

    @property
    def capability(self) -> str:
        raise RuntimeError("capability metadata unavailable")

    @property
    def provider(self) -> str:
        raise RuntimeError("provider metadata unavailable")

    @property
    def capabilities(self) -> list[str]:
        raise RuntimeError("capabilities metadata unavailable")

    @property
    def resource_profile(self) -> ResourceProfile:
        raise RuntimeError("resource metadata unavailable")

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class BrokenProviderMetadataTool(BaseTool):
    name = "broken_provider_metadata_tool"
    best_for = ["testing broken provider metadata isolation"]

    @property
    def capability(self) -> str:
        raise RuntimeError("capability metadata unavailable")

    @property
    def provider(self) -> str:
        raise RuntimeError("provider metadata unavailable")

    def get_status(self) -> ToolStatus:
        raise RuntimeError("status backend unavailable")

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class MalformedFallbackInfoTool:
    name = "malformed_fallback_info_tool"
    provider = None
    capability = "video_generation"
    best_for = {"reason": "not a list"}
    supports = "not a mapping"
    agent_skills = {"text-to-speech": True}

    def get_info(self) -> dict:
        raise RuntimeError("rich metadata unavailable")


class NonJsonRichInfoTool:
    def get_info(self) -> dict:
        return {
            "name": "non_json_rich_info_tool",
            "provider": PureWindowsPath(r"providers\local"),
            "capability": "video_generation",
            "stability": ToolStability.PRODUCTION,
            "best_for": [PureWindowsPath(r"docs\provider.md")],
            "supports": {"quality_score": math.inf, "price": Decimal("0.10")},
            "agent_skills": ("text-to-speech",),
        }


class MalformedSuccessfulSafeInfoTool:
    def get_info(self) -> dict:
        return {
            "name": "malformed_successful_safe_info_tool",
            "provider": "safe-info-provider",
            "capability": "video_generation",
            "best_for": {"reason": "not a list"},
            "supports": "not a mapping",
            "agent_skills": {"text-to-speech": True},
        }


class SparseSuccessfulSafeInfoTool(BaseTool):
    name = "sparse_successful_safe_info_tool"
    capability = "video_generation"
    provider = "safe_sparse_provider"
    runtime = ToolRuntime.API
    best_for = ["testing safe sparse metadata"]

    def get_info(self) -> dict:
        return {
            "name": self.name,
            "best_for": ["custom sparse safe metadata"],
        }

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class NonJsonRegistryInfoTool(BaseTool):
    name = "non_json_registry_info_tool"

    def get_info(self) -> dict:
        return {
            "name": self.name,
            "version": "0.1.0",
            "tier": "core",
            "capability": "video_generation",
            "provider": PureWindowsPath(r"providers\raw"),
            "stability": ToolStability.PRODUCTION,
            "status": "available",
            "execution_mode": "sync",
            "determinism": "deterministic",
            "runtime": "local",
            "module_path": __name__,
            "usage_location": PureWindowsPath(r"tools\raw_provider.py"),
            "dependencies": [PureWindowsPath(".env")],
            "install_instructions": "",
            "capabilities": [],
            "input_schema": {"type": "object", "properties": {}},
            "output_schema": {"type": "object", "properties": {}},
            "artifact_schema": {},
            "progress_schema": None,
            "supports": {"quality_score": math.inf, "price": Decimal("0.10")},
            "best_for": [PureWindowsPath(r"docs\provider.md")],
            "not_good_for": [],
            "provider_matrix": {"fallback": {"score": math.nan}},
            "resource_profile": {
                "cpu_cores": 1,
                "ram_mb": 512,
                "vram_mb": 0,
                "disk_mb": 100,
                "network_required": False,
            },
            "retry_policy": {
                "max_retries": 0,
                "backoff_seconds": 1.0,
                "retryable_errors": [],
            },
            "resume_support": "none",
            "idempotency_key_fields": [],
            "side_effects": [],
            "fallback": None,
            "fallback_tools": [],
            "agent_skills": ("text-to-speech",),
            "related_skills": ("text-to-speech",),
            "user_visible_verification": [b"listen"],
            "quality_score": math.inf,
            "historical_success_rate": math.nan,
            "latency_p50_seconds": -math.inf,
        }

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class NonDictInfoTool(BaseTool):
    name = "non_dict_info_tool"
    capability = "video_generation"
    provider = "non_dict_provider"
    best_for = ["testing non-dict metadata fallback"]

    def get_info(self):
        return ["not", "a", "mapping"]

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class SparseRegistryInfoTool(BaseTool):
    name = "sparse_registry_info_tool"
    capability = "video_generation"
    provider = "sparse_provider"
    best_for = ["testing sparse metadata normalization"]

    def get_info(self):
        return {
            "name": self.name,
            "capability": self.capability,
            "best_for": ["custom sparse metadata"],
        }

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class MalformedSuccessfulRegistryInfoTool(BaseTool):
    name = "malformed_successful_registry_info_tool"
    capability = "video_generation"
    provider = "malformed_success_provider"

    def get_info(self):
        return {
            "name": self.name,
            "capability": self.capability,
            "provider": self.provider,
            "status": ToolStatus.AVAILABLE.value,
            "dependencies": {"env:SHOULD_NOT_BECOME_DEPENDENCY": True},
            "capabilities": "single-capability",
            "input_schema": "not a schema",
            "output_schema": ["not", "a", "schema"],
            "artifact_schema": 456,
            "supports": "not a mapping",
            "best_for": "single reason",
            "not_good_for": {"reason": "not a list"},
            "provider_matrix": ["not", "a", "mapping"],
            "idempotency_key_fields": object(),
            "side_effects": None,
            "fallback_tools": {"fallback_tool": True},
            "agent_skills": "text-to-speech",
            "related_skills": object(),
            "user_visible_verification": 321,
        }

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


class MalformedRegistryFallbackCollectionsTool(BaseTool):
    name = "malformed_registry_fallback_collections_tool"
    capability = "video_generation"
    provider = "malformed_provider"
    dependencies = object()
    capabilities = object()
    input_schema = object()
    output_schema = object()
    artifact_schema = object()
    supports = object()
    best_for = object()
    not_good_for = object()
    provider_matrix = object()
    idempotency_key_fields = object()
    side_effects = object()
    fallback_tools = object()
    agent_skills = object()
    user_visible_verification = object()

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE

    def get_info(self):
        raise RuntimeError("rich metadata unavailable")

    def execute(self, inputs: dict) -> ToolResult:
        return ToolResult(success=True)


def test_binary_dependency_prefix_is_checked_like_command():
    assert BinaryDependencyTool().get_status() == ToolStatus.UNAVAILABLE


def test_base_dry_run_would_execute_reflects_dependency_status():
    unavailable = BinaryDependencyTool().dry_run({})
    available = PathIdempotencyTool().dry_run({})

    assert unavailable["status"] == ToolStatus.UNAVAILABLE.value
    assert unavailable["would_execute"] is False
    assert available["status"] == ToolStatus.AVAILABLE.value
    assert available["would_execute"] is True


def test_base_dry_run_replaces_non_finite_estimates_with_nulls():
    payload = NonFiniteDryRunTool().dry_run({})

    assert payload["estimated_cost_usd"] is None
    assert payload["estimated_runtime_seconds"] is None
    json.dumps(payload, allow_nan=False)


def test_base_dry_run_reports_estimator_failures_as_degraded_preflight():
    payload = BrokenEstimateDryRunTool().dry_run({})

    assert payload["status"] == ToolStatus.DEGRADED.value
    assert payload["would_execute"] is False
    assert payload["estimated_cost_usd"] is None
    assert payload["estimated_runtime_seconds"] is None
    assert payload["estimate_errors"] == {
        "estimated_cost_usd": "RuntimeError: cost estimator unavailable",
        "estimated_runtime_seconds": "ValueError: runtime estimator unavailable",
    }
    json.dumps(payload, allow_nan=False)


def test_base_dry_run_reports_status_failures_as_degraded_preflight():
    payload = BrokenStatusTool().dry_run({})

    assert payload["status"] == ToolStatus.DEGRADED.value
    assert payload["would_execute"] is False
    assert payload["status_error"] == "RuntimeError: status backend unavailable"
    json.dumps(payload, allow_nan=False)


def test_get_info_reports_status_failures_as_degraded_snapshot():
    info = BrokenStatusTool().get_info()

    assert info["name"] == "broken_status_tool"
    assert info["capability"] == "video_generation"
    assert info["provider"] == "broken_provider"
    assert info["status"] == ToolStatus.DEGRADED.value
    assert info["status_error"] == "RuntimeError: status backend unavailable"
    json.dumps(info, allow_nan=False)


def test_string_status_tools_remain_available_across_preflight_reports():
    tool = StringStatusTool()

    dry_run = tool.dry_run({})
    info = tool.get_info()

    assert dry_run["status"] == ToolStatus.AVAILABLE.value
    assert dry_run["would_execute"] is True
    assert info["status"] == ToolStatus.AVAILABLE.value

    registry = ToolRegistry()
    registry.register(tool)
    registry._discovered_packages.add("tools")

    assert registry.get_by_status(ToolStatus.AVAILABLE) == [tool]
    assert registry.tier_summary()["core"]["available"] == 1
    menu = registry.provider_menu()
    assert menu["video_generation"]["available"][0]["status"] == "available"

    for payload in (dry_run, info, menu, registry.provider_menu_summary()):
        json.dumps(payload, allow_nan=False)


def test_unknown_dependency_prefix_is_unavailable():
    assert UnknownDependencyTool().get_status() == ToolStatus.UNAVAILABLE


def test_env_any_dependency_prefix_accepts_any_configured_option(monkeypatch):
    monkeypatch.delenv("VIDEO_PRODUCTION_BUDDY_PRIMARY_TEST_KEY", raising=False)
    monkeypatch.delenv("VIDEO_PRODUCTION_BUDDY_SECONDARY_TEST_KEY", raising=False)
    assert EnvAnyDependencyTool().get_status() == ToolStatus.UNAVAILABLE

    monkeypatch.setenv("VIDEO_PRODUCTION_BUDDY_SECONDARY_TEST_KEY", "secondary")
    assert EnvAnyDependencyTool().get_status() == ToolStatus.AVAILABLE


def test_api_tools_with_env_setup_instructions_declare_env_dependencies():
    registry = ToolRegistry()
    registry.discover()

    setup_markers = (
        "api key",
        "environment variable",
        "env var",
        "_key",
        "token",
        "secret",
        "credentials",
        "endpoint_url",
    )
    offenders = []
    for tool in registry._tools.values():
        install_instructions = tool.install_instructions or ""
        if "no api key" in install_instructions.lower():
            continue
        if tool.runtime != ToolRuntime.API:
            continue
        if not any(marker in install_instructions.lower() for marker in setup_markers):
            continue
        if not any(
            str(dep).startswith(("env:", "env_any:")) for dep in tool.dependencies
        ):
            offenders.append(tool.name)

    assert offenders == []


def test_alternate_api_key_providers_use_env_any_dependency_contract():
    registry = ToolRegistry()
    registry.discover()

    offenders = []
    for tool in registry._tools.values():
        env_deps = [dep for dep in tool.dependencies if str(dep).startswith("env:")]
        if len(env_deps) > 1:
            offenders.append((tool.name, env_deps))

    assert offenders == []


def test_provider_menu_setup_offers_include_env_dependencies():
    registry = ToolRegistry()
    registry.discover()

    summary = registry.provider_menu_summary()
    offenders = [
        offer["tool"]
        for offer in summary["setup_offers"]
        if not any(
            str(dep).startswith(("env:", "env_any:"))
            for dep in offer.get("dependencies", [])
        )
    ]

    assert offenders == []


def test_registry_allows_idempotent_same_class_registration():
    registry = ToolRegistry()

    registry.register(DuplicateNameToolA())
    registry.register(DuplicateNameToolA())

    assert registry.get("duplicate_name_tool").__class__ is DuplicateNameToolA


def test_registry_rejects_conflicting_duplicate_tool_names():
    registry = ToolRegistry()

    registry.register(DuplicateNameToolA())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(DuplicateNameToolB())


def test_legacy_fallback_is_mirrored_in_fallback_tools():
    registry = ToolRegistry()
    registry.discover()

    offenders = [
        (tool.name, tool.fallback)
        for tool in registry._tools.values()
        if tool.fallback and tool.fallback not in tool.fallback_tools
    ]

    assert offenders == []


def test_idempotency_key_fields_are_declared_in_input_schema():
    registry = ToolRegistry()
    registry.discover()

    offenders = []
    for tool in registry._tools.values():
        properties = (tool.input_schema or {}).get("properties") or {}
        if not properties:
            continue
        missing = [
            field
            for field in tool.idempotency_key_fields
            if field not in properties
        ]
        if missing:
            offenders.append((tool.name, missing))

    assert offenders == []


def test_path_like_inputs_are_included_in_idempotency_keys():
    registry = ToolRegistry()
    registry.discover()

    def is_path_like(field_name: str) -> bool:
        normalized = field_name.lower()
        return "path" in normalized or normalized.endswith(("_dir", "_directory"))

    offenders = []
    for tool in registry._tools.values():
        properties = (tool.input_schema or {}).get("properties") or {}
        if not properties or not tool.idempotency_key_fields:
            continue
        missing = [
            field
            for field in properties
            if is_path_like(field) and field not in tool.idempotency_key_fields
        ]
        if missing:
            offenders.append((tool.name, sorted(missing)))

    assert offenders == []


def test_output_path_tools_declare_output_path_side_effects():
    registry = ToolRegistry()
    registry.discover()

    offenders = []
    for tool in registry._tools.values():
        properties = (tool.input_schema or {}).get("properties") or {}
        if "output_path" not in properties:
            continue
        if not any("output_path" in effect for effect in tool.side_effects):
            offenders.append((tool.name, tool.side_effects))

    assert offenders == []


def test_artifact_destination_path_inputs_are_declared_in_side_effects():
    registry = ToolRegistry()
    registry.discover()

    destination_fields = {
        "metadata_path",
        "output_dir",
        "video_output_path",
        "workspace_path",
    }

    offenders = []
    for tool in registry._tools.values():
        properties = (tool.input_schema or {}).get("properties") or {}
        missing = [
            field
            for field in sorted(destination_fields & set(properties))
            if not any(field in effect for effect in tool.side_effects)
        ]
        if missing:
            offenders.append((tool.name, missing, tool.side_effects))

    assert offenders == []


def test_artifact_destination_path_inputs_are_declared_in_output_schema():
    registry = ToolRegistry()
    registry.discover()

    destination_fields = {
        "metadata_path",
        "output_dir",
        "video_output_path",
        "workspace_path",
    }

    offenders = []
    for tool in registry._tools.values():
        input_properties = (tool.input_schema or {}).get("properties") or {}
        output_properties = (tool.output_schema or {}).get("properties") or {}
        missing = sorted(
            destination_fields
            & set(input_properties)
            - set(output_properties)
        )
        if missing:
            offenders.append((tool.name, missing))

    assert offenders == []


def test_input_dependent_deterministic_tools_declare_idempotency_fields():
    registry = ToolRegistry()
    registry.discover()

    offenders = []
    for tool in registry._tools.values():
        properties = (tool.input_schema or {}).get("properties") or {}
        if not properties:
            continue
        if tool.determinism.value != "deterministic":
            continue
        if not tool.idempotency_key_fields:
            offenders.append(tool.name)

    assert offenders == []


def test_idempotency_key_accepts_path_values_like_string_paths():
    tool = PathIdempotencyTool()
    expected = tool.idempotency_key(
        {"output_path": "projects/demo/renders/final.mp4"}
    )

    assert tool.idempotency_key(
        {"output_path": Path("projects/demo/renders/final.mp4")}
    ) == expected
    assert tool.idempotency_key(
        {"output_path": PureWindowsPath(r"projects\demo\renders\final.mp4")}
    ) == expected


def test_idempotency_key_canonicalizes_nested_non_json_payloads():
    tool = ComplexIdempotencyTool()

    assert tool.idempotency_key(
        {
            "payload": {
                PureWindowsPath(r"assets\reference.png"): Decimal("1.25"),
                "clip": {
                    1: "one",
                    "two": PureWindowsPath(r"renders\out.mp4"),
                },
            }
        }
    ) == tool.idempotency_key(
        {
            "payload": {
                "assets/reference.png": "1.25",
                "clip": {"1": "one", "two": "renders/out.mp4"},
            }
        }
    )


def test_tool_result_to_dict_is_json_safe_for_path_values():
    result = ToolResult(
        success=True,
        data={
            "output_path": PureWindowsPath(r"projects\demo\renders\final.mp4"),
            "nested": {
                "preview_path": PureWindowsPath(r"projects\demo\preview.mp4")
            },
        },
        artifacts=[PureWindowsPath(r"projects\demo\renders\final.mp4")],
    )

    payload = result.to_dict()

    assert payload["data"]["output_path"] == "projects/demo/renders/final.mp4"
    assert payload["data"]["nested"]["preview_path"] == "projects/demo/preview.mp4"
    assert payload["artifacts"] == ["projects/demo/renders/final.mp4"]
    json.dumps(payload, allow_nan=False)


def test_tool_result_normalizes_project_path_strings_at_output_boundary():
    result = ToolResult(
        success=True,
        data={
            "output_path": r"projects\demo\renders\final.mp4",
            "nested": [r"projects\demo\artifacts\report.json"],
            "absolute_project_path": (
                r"C:\workspace\projects\demo\renders\final.mp4"
            ),
            "input_path": r"C:\source\input.mp4",
            "message": r"keep\arbitrary\text",
        },
        artifacts=[r"projects\demo\renders\final.mp4"],
    )

    assert result.data == {
        "output_path": "projects/demo/renders/final.mp4",
        "nested": ["projects/demo/artifacts/report.json"],
        "absolute_project_path": (
            "C:/workspace/projects/demo/renders/final.mp4"
        ),
        "input_path": r"C:\source\input.mp4",
        "message": r"keep\arbitrary\text",
    }
    assert result.artifacts == ["projects/demo/renders/final.mp4"]


def test_tool_result_normalizes_project_paths_embedded_in_errors():
    result = ToolResult(
        success=False,
        error=r"Expected output was not created: projects\demo\renders\final.mp4",
    )

    assert result.error == (
        "Expected output was not created: projects/demo/renders/final.mp4"
    )


def test_base_tool_normalizes_project_references_added_after_result_creation():
    class MutatingResultTool(BaseTool):
        def execute(self, inputs: dict[str, Any]) -> ToolResult:
            result = ToolResult(success=True)
            result.data["output_path"] = r"projects\demo\renders\final.mp4"
            result.artifacts.append(r"projects\demo\renders\final.mp4")
            return result

    result = MutatingResultTool().execute({})

    assert result.data["output_path"] == "projects/demo/renders/final.mp4"
    assert result.artifacts == ["projects/demo/renders/final.mp4"]


def test_tool_result_to_dict_replaces_non_finite_numbers_with_nulls():
    result = ToolResult(
        success=True,
        data={
            "score": math.nan,
            "nested": [math.inf, -math.inf],
        },
        cost_usd=math.inf,
        duration_seconds=math.nan,
    )

    payload = result.to_dict()

    assert payload["data"]["score"] is None
    assert payload["data"]["nested"] == [None, None]
    assert payload["cost_usd"] is None
    assert payload["duration_seconds"] is None
    json.dumps(payload, allow_nan=False)


def test_tool_result_to_dict_serializes_non_finite_mapping_keys():
    result = ToolResult(
        success=True,
        data={
            "score_by_threshold": {
                math.nan: "unknown",
                math.inf: "unbounded",
                -math.inf: "below_floor",
            },
        },
    )

    payload = result.to_dict()

    assert payload["data"]["score_by_threshold"] == {
        "nan": "unknown",
        "inf": "unbounded",
        "-inf": "below_floor",
    }
    json.dumps(payload, allow_nan=False)


def test_tool_result_to_dict_serializes_enums_and_temporal_values():
    result = ToolResult(
        success=True,
        data={
            "status": ToolStatus.AVAILABLE,
            "runtime": ToolRuntime.LOCAL,
            "local_enum": LocalResultEnum.READY,
            "created_at": datetime(2026, 6, 11, 12, 30, tzinfo=timezone.utc),
            "business_day": date(2026, 6, 11),
        },
    )

    payload = result.to_dict()

    assert payload["data"]["status"] == "available"
    assert payload["data"]["runtime"] == "local"
    assert payload["data"]["local_enum"] == "ready"
    assert payload["data"]["created_at"] == "2026-06-11T12:30:00+00:00"
    assert payload["data"]["business_day"] == "2026-06-11"
    json.dumps(payload, allow_nan=False)


def test_tool_result_to_dict_serializes_uuid_and_pure_path_values():
    result = ToolResult(
        success=True,
        data={
            "asset_id": UUID("12345678-1234-5678-1234-567812345678"),
            "relative_path": PurePosixPath("projects/demo/renders/final.mp4"),
            "windows_drive_path": PureWindowsPath(r"C:\projects\demo\final.mp4"),
        },
    )

    payload = result.to_dict()

    assert payload["data"]["asset_id"] == "12345678-1234-5678-1234-567812345678"
    assert payload["data"]["relative_path"] == "projects/demo/renders/final.mp4"
    assert payload["data"]["windows_drive_path"] == "C:/projects/demo/final.mp4"
    json.dumps(payload, allow_nan=False)


def test_tool_result_to_dict_serializes_common_non_json_scalars():
    result = ToolResult(
        success=True,
        data={
            "decimal": Decimal("1.25"),
            "fraction": Fraction(1, 3),
            "bytes": b"abc",
            "bytearray": bytearray(b"xy"),
        },
    )

    payload = result.to_dict()

    assert payload["data"]["decimal"] == "1.25"
    assert payload["data"]["fraction"] == "1/3"
    assert payload["data"]["bytes"] == {
        "encoding": "base64",
        "data": "YWJj",
        "size_bytes": 3,
    }
    assert payload["data"]["bytearray"] == {
        "encoding": "base64",
        "data": "eHk=",
        "size_bytes": 2,
    }
    json.dumps(payload, allow_nan=False)


def test_tool_result_to_dict_serializes_unknown_objects_without_changing_primitives():
    result = ToolResult(
        success=True,
        data={
            "custom": CustomResultPayload(),
            "count": 7,
            "ratio": 0.25,
            "enabled": True,
            "label": "ready",
        },
    )

    payload = result.to_dict()

    assert payload["data"] == {
        "custom": "custom-result-payload",
        "count": 7,
        "ratio": 0.25,
        "enabled": True,
        "label": "ready",
    }
    json.dumps(payload, allow_nan=False)


def test_default_mutable_contract_metadata_is_instance_isolated():
    first = DefaultMutableContractToolA()
    second = DefaultMutableContractToolB()

    first.input_schema["properties"]["leaked"] = {"type": "string"}
    first.fallback_tools.append("fallback_tool")

    assert "leaked" not in second.input_schema["properties"]
    assert second.fallback_tools == []


def test_get_info_returns_snapshot_not_live_mutable_metadata():
    tool = PathIdempotencyTool()
    info = tool.get_info()

    info["input_schema"]["properties"]["leaked"] = {"type": "string"}
    info["dependencies"].append("cmd:leaked")
    info["resource_profile"]["cpu_cores"] = 99

    assert "leaked" not in tool.input_schema["properties"]
    assert tool.dependencies == []
    assert tool.resource_profile.cpu_cores == 1


def test_get_info_exposes_retry_progress_and_idempotency_contracts():
    tool = PathIdempotencyTool()
    info = tool.get_info()

    assert info["idempotency_key_fields"] == ["output_path"]
    assert info["progress_schema"] == {
        "type": "object",
        "properties": {"percent": {"type": "number"}},
    }
    assert info["retry_policy"] == {
        "max_retries": 2,
        "backoff_seconds": 3.0,
        "retryable_errors": ["timeout"],
    }

    info["retry_policy"]["retryable_errors"].append("leaked")
    assert tool.retry_policy.retryable_errors == ["timeout"]


def test_get_info_replaces_non_finite_telemetry_with_nulls():
    info = NonFiniteTelemetryTool().get_info()

    assert info["quality_score"] is None
    assert info["historical_success_rate"] is None
    assert info["latency_p50_seconds"] is None
    json.dumps(info, allow_nan=False)


def test_provider_menu_returns_snapshot_not_live_tool_metadata():
    registry = ToolRegistry()
    tool = EnvAnySetupTool()
    registry.register(tool)
    registry._discovered_packages.add("tools")

    menu = registry.provider_menu()
    entry = menu["video_generation"]["unavailable"][0]
    entry["dependencies"].append("env:LEAKED_PROVIDER_KEY")

    assert tool.dependencies == [
        "env_any:VIDEO_PRODUCTION_BUDDY_PRIMARY_TEST_KEY,VIDEO_PRODUCTION_BUDDY_SECONDARY_TEST_KEY"
    ]


def test_registry_preflight_reports_broken_tool_status_without_crashing():
    registry = ToolRegistry()
    registry.register(BrokenStatusTool())
    registry._discovered_packages.add("tools")

    envelope = registry.support_envelope()
    menu = registry.provider_menu()
    summary = registry.provider_menu_summary()
    tier_summary = registry.tier_summary()

    assert envelope["broken_status_tool"]["status"] == ToolStatus.DEGRADED.value
    assert envelope["broken_status_tool"]["status_error"] == (
        "RuntimeError: status backend unavailable"
    )
    entry = menu["video_generation"]["unavailable"][0]
    assert entry["name"] == "broken_status_tool"
    assert entry["status"] == ToolStatus.DEGRADED.value
    assert entry["status_error"] == "RuntimeError: status backend unavailable"
    assert summary["capabilities"] == [
        {
            "capability": "video_generation",
            "configured": 0,
            "total": 1,
            "available_providers": [],
            "unavailable_providers": ["broken_provider"],
        }
    ]
    assert tier_summary["core"]["degraded"] == 1
    assert registry.get_by_status(ToolStatus.DEGRADED) == [registry.get("broken_status_tool")]


def test_registry_degraded_fallback_metadata_is_strict_json_safe():
    registry = ToolRegistry()
    registry.register(NonJsonBrokenStatusTool())
    registry._discovered_packages.add("tools")

    payloads = [
        registry.support_envelope(),
        registry.capability_catalog(),
        registry.provider_catalog(),
        registry.provider_menu(),
        registry.provider_menu_summary(),
    ]

    for payload in payloads:
        json.dumps(payload, allow_nan=False)

    info = payloads[0]["non_json_broken_status_tool"]
    assert info["best_for"] == ["docs/provider.md", "1.25"]
    assert info["dependencies"] == [".env"]
    assert info["provider_matrix"]["fallback"]["score"] is None
    assert info["quality_score"] is None
    assert info["user_visible_verification"] == [
        {"encoding": "base64", "data": "bGlzdGVu", "size_bytes": 6}
    ]


def test_registry_successful_custom_info_metadata_is_strict_json_safe():
    registry = ToolRegistry()
    registry.register(NonJsonRegistryInfoTool())
    registry._discovered_packages.add("tools")

    payloads = [
        registry.support_envelope(),
        registry.capability_catalog(),
        registry.provider_catalog(),
        registry.provider_menu(),
        registry.provider_menu_summary(),
    ]

    for payload in payloads:
        json.dumps(payload, allow_nan=False)

    info = payloads[0]["non_json_registry_info_tool"]
    assert info["provider"] == "providers/raw"
    assert info["usage_location"] == "tools/raw_provider.py"
    assert info["dependencies"] == [".env"]
    assert info["supports"] == {"quality_score": None, "price": "0.10"}
    assert info["provider_matrix"]["fallback"]["score"] is None
    assert info["agent_skills"] == ["text-to-speech"]
    assert info["quality_score"] is None
    assert info["historical_success_rate"] is None
    assert info["latency_p50_seconds"] is None
    assert payloads[3]["video_generation"]["available"][0]["provider"] == "providers/raw"


def test_registry_uses_fallback_when_custom_info_is_not_mapping():
    registry = ToolRegistry()
    registry.register(NonDictInfoTool())
    registry._discovered_packages.add("tools")

    payloads = [
        registry.support_envelope(),
        registry.capability_catalog(),
        registry.provider_catalog(),
        registry.provider_menu(),
        registry.provider_menu_summary(),
    ]

    for payload in payloads:
        json.dumps(payload, allow_nan=False)

    info = payloads[0]["non_dict_info_tool"]
    assert info["name"] == "non_dict_info_tool"
    assert info["capability"] == "video_generation"
    assert info["provider"] == "non_dict_provider"
    assert info["status"] == ToolStatus.AVAILABLE.value
    assert info["status_error"] == "TypeError: get_info returned list, expected dict"
    assert payloads[3]["video_generation"]["available"][0]["name"] == "non_dict_info_tool"
    assert payloads[3]["video_generation"]["available"][0]["status_error"] == (
        "TypeError: get_info returned list, expected dict"
    )


def test_registry_normalizes_sparse_custom_info_before_reporting():
    registry = ToolRegistry()
    registry.register(SparseRegistryInfoTool())
    registry._discovered_packages.add("tools")

    payloads = [
        registry.support_envelope(),
        registry.capability_catalog(),
        registry.provider_catalog(),
        registry.provider_menu(),
        registry.provider_menu_summary(),
    ]

    for payload in payloads:
        json.dumps(payload, allow_nan=False)

    info = payloads[0]["sparse_registry_info_tool"]
    assert info["name"] == "sparse_registry_info_tool"
    assert info["capability"] == "video_generation"
    assert info["provider"] == "sparse_provider"
    assert info["status"] == ToolStatus.AVAILABLE.value
    assert info["status_error"] is None
    assert info["runtime"] == ToolRuntime.LOCAL.value
    assert info["dependencies"] == []
    assert info["install_instructions"] == ""
    assert info["best_for"] == ["custom sparse metadata"]
    assert payloads[3]["video_generation"]["available"][0]["provider"] == "sparse_provider"


def test_registry_normalizes_malformed_successful_custom_info_shapes():
    registry = ToolRegistry()
    registry.register(MalformedSuccessfulRegistryInfoTool())
    registry._discovered_packages.add("tools")

    payloads = [
        registry.support_envelope(),
        registry.capability_catalog(),
        registry.provider_catalog(),
        registry.provider_menu(),
        registry.provider_menu_summary(),
    ]

    for payload in payloads:
        json.dumps(payload, allow_nan=False)

    info = payloads[0]["malformed_successful_registry_info_tool"]
    assert info["dependencies"] == []
    assert info["capabilities"] == ["single-capability"]
    assert info["input_schema"] == {}
    assert info["output_schema"] == {}
    assert info["artifact_schema"] == {}
    assert info["supports"] == {}
    assert info["best_for"] == ["single reason"]
    assert info["not_good_for"] == []
    assert info["provider_matrix"] == {}
    assert info["idempotency_key_fields"] == []
    assert info["side_effects"] == []
    assert info["fallback_tools"] == []
    assert info["agent_skills"] == ["text-to-speech"]
    assert info["related_skills"] == []
    assert info["user_visible_verification"] == []
    entry = payloads[3]["video_generation"]["available"][0]
    assert entry["best_for"] == ["single reason"]
    assert entry["dependencies"] == []
    assert entry["provider_matrix"] == {}


def test_registry_fallback_tolerates_malformed_collection_metadata():
    registry = ToolRegistry()
    registry.register(MalformedRegistryFallbackCollectionsTool())
    registry._discovered_packages.add("tools")

    payloads = [
        registry.support_envelope(),
        registry.capability_catalog(),
        registry.provider_catalog(),
        registry.provider_menu(),
        registry.provider_menu_summary(),
    ]

    for payload in payloads:
        json.dumps(payload, allow_nan=False)

    info = payloads[0]["malformed_registry_fallback_collections_tool"]
    assert info["name"] == "malformed_registry_fallback_collections_tool"
    assert info["capability"] == "video_generation"
    assert info["provider"] == "malformed_provider"
    assert info["status"] == ToolStatus.AVAILABLE.value
    assert info["status_error"] == "RuntimeError: rich metadata unavailable"
    assert info["dependencies"] == []
    assert info["capabilities"] == []
    assert info["input_schema"] == {}
    assert info["output_schema"] == {}
    assert info["artifact_schema"] == {}
    assert info["supports"] == {}
    assert info["best_for"] == []
    assert info["not_good_for"] == []
    assert info["provider_matrix"] == {}
    assert info["idempotency_key_fields"] == []
    assert info["side_effects"] == []
    assert info["fallback_tools"] == []
    assert info["agent_skills"] == []
    assert info["related_skills"] == []
    assert info["user_visible_verification"] == []
    assert payloads[3]["video_generation"]["available"][0]["name"] == (
        "malformed_registry_fallback_collections_tool"
    )


def test_registry_preflight_uses_fallback_when_provider_metadata_raises():
    registry = ToolRegistry()
    registry.register(BrokenProviderMetadataTool())
    registry._discovered_packages.add("tools")

    menu = registry.provider_menu()
    summary = registry.provider_menu_summary()

    entry = menu["generic"]["unavailable"][0]
    assert entry["name"] == "broken_provider_metadata_tool"
    assert entry["provider"] == "unknown"
    assert entry["status"] == ToolStatus.DEGRADED.value
    assert entry["status_error"] == "RuntimeError: status backend unavailable"
    assert summary["capabilities"] == [
        {
            "capability": "generic",
            "configured": 0,
            "total": 1,
            "available_providers": [],
            "unavailable_providers": ["unknown"],
        }
    ]
    json.dumps(menu, allow_nan=False)
    json.dumps(summary, allow_nan=False)


def test_safe_tool_info_fallback_tolerates_malformed_collection_metadata():
    info = safe_tool_info(MalformedFallbackInfoTool())

    assert info["name"] == "malformed_fallback_info_tool"
    assert info["provider"] == "unknown"
    assert info["capability"] == "video_generation"
    assert info["best_for"] == []
    assert info["supports"] == {}
    assert info["agent_skills"] == []
    json.dumps(info, allow_nan=False)


def test_safe_tool_info_sanitizes_successful_rich_info_payload():
    info = safe_tool_info(NonJsonRichInfoTool())

    assert info["name"] == "non_json_rich_info_tool"
    assert info["provider"] == "providers/local"
    assert info["capability"] == "video_generation"
    assert info["stability"] == "production"
    assert info["best_for"] == ["docs/provider.md"]
    assert info["supports"] == {"quality_score": None, "price": "0.10"}
    assert info["agent_skills"] == ["text-to-speech"]
    json.dumps(info, allow_nan=False)


def test_safe_tool_info_normalizes_successful_malformed_info_shapes():
    info = safe_tool_info(MalformedSuccessfulSafeInfoTool())

    assert info["name"] == "malformed_successful_safe_info_tool"
    assert info["provider"] == "safe-info-provider"
    assert info["capability"] == "video_generation"
    assert info["best_for"] == []
    assert info["supports"] == {}
    assert info["agent_skills"] == []
    json.dumps(info, allow_nan=False)


def test_safe_tool_info_merges_sparse_successful_info_over_safe_defaults():
    info = safe_tool_info(SparseSuccessfulSafeInfoTool())

    assert info["name"] == "sparse_successful_safe_info_tool"
    assert info["provider"] == "safe_sparse_provider"
    assert info["capability"] == "video_generation"
    assert info["runtime"] == ToolRuntime.API.value
    assert info["best_for"] == ["custom sparse safe metadata"]
    assert info["supports"] == {}
    assert info["agent_skills"] == []
    json.dumps(info, allow_nan=False)


def test_registry_query_helpers_skip_tools_with_broken_metadata():
    registry = ToolRegistry()
    safe_tool = QueryableMetadataTool()
    registry.register(safe_tool)
    registry.register(BrokenRegistryQueryMetadataTool())

    assert registry.get_by_tier(ToolTier.VOICE) == [safe_tool]
    assert registry.get_by_capability("query_capability") == [safe_tool]
    assert registry.get_by_provider("query_provider") == [safe_tool]
    assert registry.get_by_stability(ToolStability.PRODUCTION) == [safe_tool]
    assert registry.find_by_capability("query_alias") == [safe_tool]
    assert registry.gpu_required_tools() == ["queryable_metadata_tool"]
    assert registry.network_required_tools() == ["queryable_metadata_tool"]


def test_output_schema_required_fields_are_declared_as_properties():
    registry = ToolRegistry()
    registry.discover()

    offenders = []
    for tool in registry._tools.values():
        required = tool.output_schema.get("required", [])
        if not required:
            continue
        properties = (tool.output_schema or {}).get("properties") or {}
        missing = [
            field
            for field in required
            if field not in properties
        ]
        if missing:
            offenders.append((tool.name, missing))

    assert offenders == []


def test_discovered_tools_declare_non_empty_output_schema_properties():
    registry = ToolRegistry()
    registry.discover()

    offenders = [
        tool.name
        for tool in registry._tools.values()
        if not ((tool.output_schema or {}).get("properties") or {})
    ]

    assert offenders == []


def test_literal_tool_result_data_keys_are_declared_in_output_schema():
    registry = ToolRegistry()
    registry.discover()
    tools_by_class = {
        (tool.__class__.__module__, tool.__class__.__name__): tool
        for tool in registry._tools.values()
    }

    offenders = []
    for path in sorted(Path("tools").rglob("*.py")):
        if path.name.startswith("__"):
            continue
        module = ".".join(path.with_suffix("").parts)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for class_node in [
            node for node in tree.body if isinstance(node, ast.ClassDef)
        ]:
            tool = tools_by_class.get((module, class_node.name))
            if tool is None:
                continue
            declared = set(((tool.output_schema or {}).get("properties") or {}).keys())
            literal_data_keys = set()
            for node in ast.walk(class_node):
                if not isinstance(node, ast.Call):
                    continue
                function_name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else ""
                )
                if function_name != "ToolResult":
                    continue
                for keyword in node.keywords:
                    if keyword.arg != "data" or not isinstance(keyword.value, ast.Dict):
                        continue
                    for key in keyword.value.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            literal_data_keys.add(key.value)
            missing = sorted(literal_data_keys - declared)
            if missing:
                offenders.append((tool.name, missing))

    assert offenders == []


def test_discovered_tools_declare_non_empty_input_schema_properties():
    registry = ToolRegistry()
    registry.discover()

    offenders = [
        tool.name
        for tool in registry._tools.values()
        if not ((tool.input_schema or {}).get("properties") or {})
    ]

    assert offenders == []


def test_output_path_tools_declare_output_path_in_output_schema():
    registry = ToolRegistry()
    registry.discover()

    offenders = []
    for tool in registry._tools.values():
        input_properties = (tool.input_schema or {}).get("properties") or {}
        if "output_path" not in input_properties:
            continue
        output_schema = tool.output_schema or {}
        output_properties = output_schema.get("properties") or {}
        if output_properties and "output_path" not in output_properties:
            offenders.append(tool.name)

    assert offenders == []


def test_output_schemas_declare_required_success_payload_shape():
    registry = ToolRegistry()
    registry.discover()

    shape_keywords = {"required", "anyOf", "oneOf", "allOf"}
    offenders = []
    for tool in registry._tools.values():
        output_schema = tool.output_schema or {}
        output_properties = output_schema.get("properties") or {}
        if output_properties and not (shape_keywords & set(output_schema)):
            offenders.append(tool.name)

    assert offenders == []


def test_tool_input_and_output_schemas_are_json_objects():
    registry = ToolRegistry()
    registry.discover()

    offenders = []
    for tool in registry._tools.values():
        for schema_name in ("input_schema", "output_schema"):
            schema = getattr(tool, schema_name) or {}
            if schema.get("type") != "object":
                offenders.append((tool.name, schema_name, schema.get("type")))

    assert offenders == []


def test_tool_input_and_output_schemas_are_meta_schema_valid():
    registry = ToolRegistry()
    registry.discover()

    offenders = []
    for tool in registry._tools.values():
        for schema_name in ("input_schema", "output_schema"):
            schema = getattr(tool, schema_name) or {}
            try:
                validator_for(schema).check_schema(schema)
            except Exception as exc:
                offenders.append((tool.name, schema_name, f"{type(exc).__name__}: {exc}"))

    assert offenders == []


def test_routed_generation_and_selector_tools_declare_agent_skills():
    registry = ToolRegistry()
    registry.discover()

    routed_capabilities = {"image_generation", "video_generation", "tts"}
    routed_runtimes = {ToolRuntime.API, ToolRuntime.HYBRID, ToolRuntime.LOCAL_GPU}
    offenders = []
    for tool in registry._tools.values():
        routed_generation_provider = (
            tool.capability in {"image_generation", "video_generation"}
            and tool.runtime in routed_runtimes
            and tool.tier in {ToolTier.GENERATE, ToolTier.SOURCE}
        )
        selector = tool.capability in routed_capabilities and tool.provider == "selector"
        if (routed_generation_provider or selector) and not tool.agent_skills:
            offenders.append((tool.name, tool.capability, tool.provider))

    assert offenders == []


def test_api_and_hybrid_source_generation_tools_declare_agent_skills():
    registry = ToolRegistry()
    registry.discover()

    routed_tiers = {ToolTier.SOURCE, ToolTier.GENERATE, ToolTier.VOICE}
    routed_runtimes = {ToolRuntime.API, ToolRuntime.HYBRID}
    offenders = [
        (tool.name, tool.tier.value, tool.capability, tool.provider, tool.runtime.value)
        for tool in registry._tools.values()
        if tool.tier in routed_tiers
        and tool.runtime in routed_runtimes
        and not tool.agent_skills
    ]

    assert sorted(offenders) == []


def test_provider_menu_setup_offers_exclude_local_runtime_setup():
    registry = ToolRegistry()
    registry.discover()

    summary = registry.provider_menu_summary()
    local_runtime_offers = [
        offer["tool"]
        for offer in summary["setup_offers"]
        if registry.get(offer["tool"]).runtime
        not in {ToolRuntime.API, ToolRuntime.HYBRID}
    ]

    assert local_runtime_offers == []


def test_provider_menu_setup_offers_include_env_any_dependencies(monkeypatch):
    monkeypatch.delenv("VIDEO_PRODUCTION_BUDDY_PRIMARY_TEST_KEY", raising=False)
    monkeypatch.delenv("VIDEO_PRODUCTION_BUDDY_SECONDARY_TEST_KEY", raising=False)
    registry = ToolRegistry()
    registry.register(EnvAnySetupTool())
    registry._discovered_packages.add("tools")

    summary = registry.provider_menu_summary()

    assert summary["setup_offers"] == [
        {
            "capability": "video_generation",
            "tool": "env_any_setup_tool",
            "provider": "env_any_provider",
            "install_instructions": (
                "Set VIDEO_PRODUCTION_BUDDY_PRIMARY_TEST_KEY or "
                "VIDEO_PRODUCTION_BUDDY_SECONDARY_TEST_KEY to your API key."
            ),
            "dependencies": [
                "env_any:VIDEO_PRODUCTION_BUDDY_PRIMARY_TEST_KEY,VIDEO_PRODUCTION_BUDDY_SECONDARY_TEST_KEY"
            ],
        }
    ]


def test_tools_declare_best_for_provider_menu_context():
    registry = ToolRegistry()
    registry.discover()

    offenders = [
        tool.name
        for tool in registry._tools.values()
        if not tool.best_for
    ]

    assert offenders == []


def test_hybrid_tools_explain_setup_path():
    registry = ToolRegistry()
    registry.discover()

    offenders = [
        tool.name
        for tool in registry._tools.values()
        if tool.runtime == ToolRuntime.HYBRID and not tool.install_instructions
    ]

    assert offenders == []
