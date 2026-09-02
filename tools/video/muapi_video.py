"""MuAPI video generation through the hosted submit -> poll API."""

from __future__ import annotations

import ipaddress
import os
import time
from typing import Any
from urllib.parse import urlsplit

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)
from tools.video._shared import (
    probe_output,
    require_generated_video_output_path,
    validate_video_operation,
)

_DEFAULT_BASE_URL = "https://api.muapi.ai/api/v1"
_TEXT_ENDPOINT = "seedance-2-text-to-video"
_TEXT_FAST_ENDPOINT = "seedance-2-text-to-video-fast"
_IMAGE_ENDPOINT = "seedance-2-image-to-video"
_IMAGE_FAST_ENDPOINT = "seedance-2-image-to-video-fast"
_OPERATIONS = {"text_to_video", "image_to_video"}
_DEFAULT_ENDPOINTS = {
    "text_to_video": _TEXT_ENDPOINT,
    "image_to_video": _IMAGE_ENDPOINT,
}
_ENDPOINTS: dict[str, dict[str, Any]] = {
    _TEXT_ENDPOINT: {
        "operation": "text_to_video",
        "name": "Seedance 2 Text-to-Video",
        "speed": "medium",
        "quality": "high",
        "cost_per_second": 0.25,
    },
    _TEXT_FAST_ENDPOINT: {
        "operation": "text_to_video",
        "name": "Seedance 2 Fast Text-to-Video",
        "speed": "fast",
        "quality": "high",
        "cost_per_second": 0.15,
    },
    _IMAGE_ENDPOINT: {
        "operation": "image_to_video",
        "name": "Seedance 2 Image-to-Video",
        "speed": "medium",
        "quality": "high",
        "cost_per_second": 0.25,
    },
    _IMAGE_FAST_ENDPOINT: {
        "operation": "image_to_video",
        "name": "Seedance 2 Fast Image-to-Video",
        "speed": "fast",
        "quality": "high",
        "cost_per_second": 0.15,
    },
}
_ASPECT_RATIOS = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]
_MODEL_SOURCE_URL = "https://muapi.ai/ai-video-api"


def _api_key() -> str | None:
    return os.environ.get("MUAPI_API_KEY") or os.environ.get("MU_API_KEY")


def _base_url() -> str:
    return os.environ.get("MUAPI_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


def _prediction_id(payload: dict[str, Any]) -> str | None:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    for key in ("request_id", "id", "prediction_id", "task_id"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _status_payload(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, dict) and "status" in data:
        return data
    return payload


def _collect_video_urls(value: Any) -> list[str]:
    """Collect output URLs without walking request/input fields."""

    if isinstance(value, str):
        return [value] if value.startswith(("http://", "https://")) else []
    if isinstance(value, list):
        urls: list[str] = []
        for item in value:
            urls.extend(_collect_video_urls(item))
        return urls
    if not isinstance(value, dict):
        return []

    urls: list[str] = []
    for key in ("video_url", "video", "output_url", "download_url", "url"):
        if key in value:
            urls.extend(_collect_video_urls(value[key]))
    for key in ("content", "outputs", "output", "result", "results", "data", "output_data"):
        if key in value:
            urls.extend(_collect_video_urls(value[key]))
    return urls


def _safe_output_url(url: str) -> str:
    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not hostname:
        raise ValueError("MuAPI returned a non-HTTPS output URL")
    if parsed.username or parsed.password or parsed.port:
        raise ValueError("MuAPI returned an output URL with credentials or a port")
    if hostname in {"localhost", "localhost.localdomain"}:
        raise ValueError("MuAPI returned a local output URL")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and (address.is_private or address.is_loopback or address.is_link_local):
        raise ValueError("MuAPI returned a private output URL")
    return url


class MuapiVideo(BaseTool):
    name = "muapi_video"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "muapi"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = ["env_any:MUAPI_API_KEY,MU_API_KEY"]
    install_instructions = (
        "Set MUAPI_API_KEY to your MuAPI API key (MU_API_KEY is also accepted).\n"
        "  Get one at https://muapi.ai/keys and see https://muapi.ai/docs"
    )
    agent_skills = ["ai-video-gen"]

    capabilities = ["text_to_video", "image_to_video"]
    supports = {
        "text_to_video": True,
        "image_to_video": True,
        "aspect_ratio": True,
        "duration": True,
        "native_audio": True,
    }
    best_for = [
        "hosted video generation through one MuAPI key",
        "Seedance 2 text-to-video and image-to-video workflows",
        "projects that need an async submit, poll, and local-download lifecycle",
    ]
    not_good_for = ["offline generation", "long-form video rendering"]
    fallback_tools = ["seedance_video", "wan_video_api", "kling_video"]
    quality_score = 0.9
    model_options = [
        {
            "id": endpoint,
            "name": meta["name"],
            "field": "model_variant",
            "operation": meta["operation"],
            "default_for_operations": [
                operation
                for operation, default_endpoint in _DEFAULT_ENDPOINTS.items()
                if endpoint == default_endpoint
            ],
            "quality": meta["quality"],
            "speed": meta["speed"],
            "source_url": _MODEL_SOURCE_URL,
        }
        for endpoint, meta in _ENDPOINTS.items()
    ]

    input_schema = {
        "type": "object",
        "required": ["prompt", "output_path"],
        "properties": {
            "prompt": {"type": "string"},
            "operation": {
                "type": "string",
                "enum": sorted(_OPERATIONS),
                "default": "text_to_video",
            },
            "model_variant": {
                "type": "string",
                "enum": list(_ENDPOINTS),
                "description": "MuAPI model endpoint. Omit to use the operation default.",
            },
            "duration": {
                "type": "integer",
                "minimum": 4,
                "maximum": 15,
                "default": 5,
            },
            "aspect_ratio": {
                "type": "string",
                "enum": _ASPECT_RATIOS,
                "default": "16:9",
            },
            "image_url": {
                "type": "string",
                "description": "Start-frame image URL for image_to_video.",
            },
            "output_path": {"type": "string"},
        },
    }
    output_schema = {
        "type": "object",
        "required": [
            "provider",
            "model",
            "prompt",
            "operation",
            "duration",
            "aspect_ratio",
            "prediction_id",
            "output",
            "output_path",
            "format",
            "file_size_bytes",
        ],
        "properties": {
            "provider": {"type": "string", "const": "muapi"},
            "model": {"type": "string"},
            "prompt": {"type": "string"},
            "operation": {"type": "string", "enum": sorted(_OPERATIONS)},
            "duration": {"type": "integer", "minimum": 4, "maximum": 15},
            "aspect_ratio": {"type": "string", "enum": _ASPECT_RATIOS},
            "prediction_id": {"type": "string"},
            "output": {"type": "string"},
            "output_path": {"type": "string"},
            "format": {"type": "string", "const": "mp4"},
            "file_size_bytes": {"type": "integer", "minimum": 0},
            "duration_seconds": {"type": "number", "minimum": 0},
            "file_size_mb": {"type": "number", "minimum": 0},
            "video_width": {"type": "integer", "minimum": 0},
            "video_height": {"type": "integer", "minimum": 0},
            "video_codec": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=500, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = [
        "prompt",
        "output_path",
        "model_variant",
        "operation",
        "duration",
        "aspect_ratio",
        "image_url",
    ]
    side_effects = ["writes video file to output_path", "calls MuAPI video API"]
    user_visible_verification = [
        "Inspect sampled frames for motion coherence and visual quality"
    ]

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if _api_key() else ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        operation = inputs.get("operation", "text_to_video")
        model = inputs.get("model_variant") or _DEFAULT_ENDPOINTS.get(operation, _TEXT_ENDPOINT)
        spec = _ENDPOINTS.get(str(model), _ENDPOINTS[_TEXT_ENDPOINT])
        try:
            duration = int(inputs.get("duration", 5))
        except (TypeError, ValueError):
            duration = 5
        return round(duration * float(spec["cost_per_second"]), 3)

    def estimate_runtime(self, _inputs: dict[str, Any]) -> float:
        return 120.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        operation = inputs.get("operation", "text_to_video")
        operation_error = validate_video_operation(operation, _OPERATIONS)
        if operation_error:
            return ToolResult(success=False, error=operation_error)

        model = inputs.get("model_variant") or _DEFAULT_ENDPOINTS[operation]
        if model not in _ENDPOINTS:
            return ToolResult(
                success=False,
                error=(
                    f"Unknown model_variant {model!r}. "
                    f"Available: {', '.join(_ENDPOINTS)}."
                ),
            )
        model_operation = _ENDPOINTS[model]["operation"]
        if model_operation != operation:
            return ToolResult(
                success=False,
                error=(
                    f"model_variant {model!r} supports {model_operation}, "
                    f"but operation is {operation}."
                ),
            )
        if operation == "image_to_video" and not inputs.get("image_url"):
            return ToolResult(success=False, error="image_to_video requires image_url")

        try:
            duration = int(inputs.get("duration", 5))
        except (TypeError, ValueError):
            return ToolResult(success=False, error="duration must be an integer from 4 to 15")
        if not 4 <= duration <= 15:
            return ToolResult(success=False, error="duration must be an integer from 4 to 15")

        output_path, output_error = require_generated_video_output_path(inputs, self.name)
        if output_error:
            return output_error

        api_key = _api_key()
        if not api_key:
            return ToolResult(
                success=False,
                error="MUAPI_API_KEY not set. " + self.install_instructions,
            )

        import requests

        start = time.time()
        payload = {
            "prompt": inputs["prompt"],
            "duration": duration,
            "aspect_ratio": inputs.get("aspect_ratio", "16:9"),
        }
        if operation == "image_to_video":
            payload["images_list"] = [inputs["image_url"]]
        headers = {"x-api-key": api_key, "Content-Type": "application/json"}

        try:
            submit = requests.post(
                f"{_base_url()}/{model}",
                headers=headers,
                json=payload,
                timeout=30,
            )
            submit.raise_for_status()
            request_id = _prediction_id(submit.json())
            if not request_id:
                return ToolResult(
                    success=False,
                    error="MuAPI video generation returned no request_id",
                )

            result_payload = self._poll_prediction(request_id, headers)
            urls = _collect_video_urls(result_payload)
            if not urls:
                return ToolResult(
                    success=False,
                    error="MuAPI video generation completed without an output URL",
                )
            download_url = _safe_output_url(urls[0])
            video_response = requests.get(
                download_url,
                timeout=300,
                allow_redirects=False,
            )
            if 300 <= getattr(video_response, "status_code", 200) < 400:
                return ToolResult(
                    success=False,
                    error="MuAPI video output redirected unexpectedly",
                )
            video_response.raise_for_status()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(video_response.content)
        except Exception as exc:
            return ToolResult(success=False, error=f"MuAPI video generation failed: {exc}")

        probed = probe_output(output_path)
        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": model,
                "prompt": inputs["prompt"],
                "operation": operation,
                "duration": payload["duration"],
                "aspect_ratio": payload["aspect_ratio"],
                "prediction_id": request_id,
                "output": str(output_path),
                "output_path": str(output_path),
                "format": "mp4",
                **probed,
            },
            artifacts=[str(output_path)],
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            model=model,
        )

    def _poll_prediction(
        self,
        request_id: str,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        import requests

        deadline = time.time() + 600
        delay = 2
        while True:
            response = requests.get(
                f"{_base_url()}/predictions/{request_id}/result",
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            data = _status_payload(payload)
            status = str(data.get("status", "")).lower()
            if status in {"completed", "succeeded", "success"}:
                return data
            if status in {"failed", "error", "cancelled", "canceled"}:
                message = data.get("error") or data.get("message") or status
                raise RuntimeError(f"MuAPI video generation {message}")
            if not status and _collect_video_urls(data):
                return data
            if time.time() >= deadline:
                raise TimeoutError("MuAPI video generation timed out")
            time.sleep(delay)
            delay = min(delay * 2, 10)
