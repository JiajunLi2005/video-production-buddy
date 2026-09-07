"""MuAPI video generation through the hosted submit -> poll API."""

from __future__ import annotations

import io
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from lib.ffmpeg_validation import STRICT_DECODE_ARGS, require_clean_decode
from tools.video._muapi_download import download_video

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolTier,
)
from tools.video._shared import (
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

    dependencies = ["env_any:MUAPI_API_KEY,MU_API_KEY", "cmd:ffmpeg", "cmd:ffprobe"]
    install_instructions = (
        "Set MUAPI_API_KEY to your MuAPI API key (MU_API_KEY is also accepted).\n"
        "  Install FFmpeg and ffprobe to validate downloads before publication.\n"
        "  Get a key at https://muapi.ai/keys; usage: docs/PROVIDERS.md#muapi"
    )
    agent_skills = ["ai-video-gen"]

    capabilities = ["text_to_video", "image_to_video"]
    supports = {
        "text_to_video": True,
        "image_to_video": True,
        "local_image_input": True,
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
            "image_path": {
                "type": "string",
                "description": "Local JPEG, PNG or WebP, up to 10 MiB; uploaded directly to MuAPI.",
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
        "image_path",
    ]
    side_effects = [
        "writes validated video file to output_path", "calls MuAPI video API",
        "uploads image_path to MuAPI storage for image_to_video",
    ]
    user_visible_verification = [
        "Inspect sampled frames for motion coherence and visual quality"
    ]

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
        if operation == "image_to_video" and not (inputs.get("image_url") or inputs.get("image_path")):
            return ToolResult(success=False, error="image_to_video requires image_url or image_path")
        if inputs.get("image_url") and inputs.get("image_path"):
            return ToolResult(success=False, error="Provide only one of image_url or image_path")

        try:
            duration = int(inputs.get("duration", 5))
        except (TypeError, ValueError):
            return ToolResult(success=False, error="duration must be an integer from 4 to 15")
        if not 4 <= duration <= 15:
            return ToolResult(success=False, error="duration must be an integer from 4 to 15")

        output_path, output_error = require_generated_video_output_path(inputs, self.name)
        if output_error:
            return output_error
        if output_path.suffix.lower() != ".mp4":
            return ToolResult(success=False, error="MuAPI output_path must use the .mp4 extension")
        if not all(shutil.which(command) for command in ("ffmpeg", "ffprobe")):
            return ToolResult(success=False, error="MuAPI needs FFmpeg and ffprobe to validate output")

        image_data = None
        if operation == "image_to_video" and inputs.get("image_path"):
            try:
                image_data = self._read_image(Path(inputs["image_path"]))
            except Exception as exc:
                return ToolResult(success=False, error=f"Invalid MuAPI reference image: {exc}")

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
        headers = {"x-api-key": api_key, "Content-Type": "application/json"}

        candidate_path = None
        try:
            if operation == "image_to_video":
                image_url = inputs.get("image_url")
                if image_data is not None:
                    uploaded = requests.post(
                        f"{_base_url()}/upload_file",
                        headers={"x-api-key": api_key},
                        files={"file": image_data},
                        timeout=60,
                        allow_redirects=False,
                    )
                    uploaded.raise_for_status()
                    if uploaded.status_code != 200:
                        raise ValueError("MuAPI image upload did not return HTTP 200")
                    image_url = uploaded.json().get("url")
                    if not isinstance(image_url, str) or urlsplit(image_url).scheme != "https":
                        raise ValueError("MuAPI image upload returned no HTTPS URL")
                payload["images_list"] = [image_url]
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
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=output_path.parent, prefix=f".{output_path.stem}-", suffix=".mp4", delete=False
            ) as candidate:
                candidate_path = Path(candidate.name)
            download_video(urls[0], candidate_path)
            probed = self._validate_download(candidate_path)
            os.replace(candidate_path, output_path)
        except Exception as exc:
            return ToolResult(success=False, error=f"MuAPI video generation failed: {exc}")
        finally:
            if candidate_path is not None:
                candidate_path.unlink(missing_ok=True)

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

    @staticmethod
    def _read_image(path: Path) -> tuple[str, bytes, str]:
        from PIL import Image

        with path.open("rb") as handle:
            content = handle.read(10 * 1024 * 1024 + 1)
        if not content or len(content) > 10 * 1024 * 1024:
            raise ValueError("Reference image must be nonempty and at most 10 MiB")
        with Image.open(io.BytesIO(content)) as image:
            formats = {"JPEG": ("jpg", "image/jpeg"), "PNG": ("png", "image/png"), "WEBP": ("webp", "image/webp")}
            if image.format not in formats:
                raise ValueError("Reference image must be JPEG, PNG or WebP")
            extension, media_type = formats[image.format]
            image.verify()
        return f"reference.{extension}", content, media_type

    def _validate_download(self, path: Path) -> dict[str, Any]:
        # Do not let an untrusted download turn probing into another network
        # request or an external-track read through a disguised playlist/MOV.
        input_options = [
            "-protocol_whitelist", "file,pipe", "-format_whitelist", "mov",
            "-enable_drefs", "0", "-use_absolute_path", "0",
        ]
        probe = self.run_command([
            "ffprobe", "-v", "level+error", *input_options,
            "-show_format", "-show_streams", "-of", "json", str(path),
        ], timeout=30)
        require_clean_decode(probe.stderr or "")
        media = json.loads(probe.stdout)
        video = next((s for s in media.get("streams", []) if s.get("codec_type") == "video"), None)
        duration = float(media.get("format", {}).get("duration", 0))
        if (
            video is None or not math.isfinite(duration) or duration <= 0
            or int(video.get("width", 0)) <= 0 or int(video.get("height", 0)) <= 0
            or "mp4" not in media.get("format", {}).get("format_name", "").split(",")
        ):
            raise ValueError("MuAPI output is not a valid, nonempty MP4 video")
        decoded = self.run_command([
            "ffmpeg", "-nostdin", "-hide_banner", "-nostats", *STRICT_DECODE_ARGS,
            *input_options, "-i", str(path), "-map", "0:v:0", "-map", "0:a?",
            "-progress", "pipe:1", "-f", "null", "-",
        ], timeout=600)
        require_clean_decode(decoded.stderr or "")
        frames = [line.removeprefix("frame=").strip()
                  for line in decoded.stdout.splitlines() if line.startswith("frame=")]
        if not any(value.isdigit() and int(value) > 0 for value in frames):
            raise ValueError("MuAPI output contains no decodable video frames")
        return {
            "file_size_bytes": path.stat().st_size,
            "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            "duration_seconds": duration,
            "video_width": int(video["width"]), "video_height": int(video["height"]),
            "video_codec": video.get("codec_name", ""),
        }

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
