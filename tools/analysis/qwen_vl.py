"""Qwen vision-language understanding via Alibaba Cloud Bailian / DashScope.

Drives the Qwen-VL / Qwen3.7 vision models over the OpenAI-compatible
Chat Completions endpoint with multimodal content (image_url / video_url).
Uses DASHSCOPE_API_KEY. Useful for reference-video analysis, frame/scene quality
checks, hallucination review, and describing what is visible in an asset.

Endpoint: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
Auth:     Authorization: Bearer DASHSCOPE_API_KEY

This is a standalone capability tool (capability=vision_understanding). The agent
calls it when a "look at this image/video" sub-task is needed; it is not
auto-wired into any pipeline stage.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

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
from tools.output_paths import portable_output_path, require_optional_project_sidecar_path

_CHAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

_MODELS: dict[str, dict[str, Any]] = {
    "qwen3-vl-plus": {
        "name": "Qwen3-VL Plus",
        "quality": "high",
        "speed": "medium",
        "note": "Balanced vision-language model (default)",
    },
    "qwen3-vl-flash": {
        "name": "Qwen3-VL Flash",
        "quality": "medium",
        "speed": "fast",
        "note": "Low-cost / high-throughput vision",
    },
    "qwen3.7-plus": {
        "name": "Qwen3.7 Plus",
        "quality": "highest",
        "speed": "medium",
        "note": "Unified multimodal agent: image + video + long context + tool calling",
    },
}


class QwenVL(BaseTool):
    name = "qwen_vl"
    version = "0.1.0"
    tier = ToolTier.ANALYZE
    capability = "vision_understanding"
    provider = "bailian"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = ["env:DASHSCOPE_API_KEY"]
    install_instructions = (
        "Set the DASHSCOPE_API_KEY environment variable:\n"
        "  export DASHSCOPE_API_KEY=your_key_here\n"
        "Get a key at https://bailian.console.aliyun.com/"
    )
    fallback_tools: list[str] = []
    agent_skills: list[str] = ["video-understand"]

    capabilities = ["image_understanding", "video_understanding", "visual_qa", "ocr"]
    supports = {
        "image_input": True,
        "video_input": True,
        "local_files": True,
        "offline": False,
    }
    best_for = [
        "Describing / analyzing a generated frame, keyframe, or product photo",
        "Reference-video analysis (what the reference is doing, pacing cues)",
        "Hallucination review — checking whether a generated asset matches the brief",
        "OCR and on-screen text extraction from frames",
    ]
    not_good_for = ["generating images (use wanx_image)", "offline analysis"]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "anyOf": [
            {"required": ["image_url"]},
            {"required": ["image_path"]},
            {"required": ["video_url"]},
        ],
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Question / instruction about the image or video (e.g. 'Describe the composition and any product logos visible.')",
            },
            "image_url": {
                "type": "string",
                "description": "Public image URL to analyze (HTTP/HTTPS)",
            },
            "image_path": {
                "type": "string",
                "description": "Local image path (auto base64-encoded as a data URL)",
            },
            "video_url": {
                "type": "string",
                "description": "Public video URL to analyze (DashScope vision models accept video URLs)",
            },
            "model": {
                "type": "string",
                "enum": list(_MODELS.keys()),
                "default": "qwen3-vl-plus",
            },
            "max_tokens": {"type": "integer", "default": 2048, "minimum": 1},
            "output_path": {
                "type": "string",
                "description": "Optional project-scoped path to write the analysis text.",
            },
        },
    }
    output_schema = {
        "type": "object",
        "required": [
            "provider",
            "model",
            "model_name",
            "text",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "output",
            "output_path",
        ],
        "properties": {
            "provider": {"type": "string"},
            "model": {"type": "string", "enum": list(_MODELS.keys())},
            "model_name": {"type": "string"},
            "text": {"type": "string"},
            "prompt_tokens": {"type": ["integer", "null"]},
            "completion_tokens": {"type": ["integer", "null"]},
            "total_tokens": {"type": ["integer", "null"]},
            "output": {"type": ["string", "null"]},
            "output_path": {"type": ["string", "null"]},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=20, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = [
        "prompt",
        "image_url",
        "image_path",
        "video_url",
        "model",
        "max_tokens",
        "output_path",
    ]
    side_effects = ["calls Bailian/DashScope multimodal Chat API", "optionally writes analysis to output_path"]
    user_visible_verification = ["Read the analysis for accuracy against the actual image/video content"]

    def _get_api_key(self) -> str | None:
        return os.environ.get("DASHSCOPE_API_KEY")

    def get_status(self) -> ToolStatus:
        if self._get_api_key():
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return round(inputs.get("max_tokens", 2048) / 1000 * 0.003, 5)

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        model = inputs.get("model", "qwen3-vl-plus")
        if model not in _MODELS:
            return ToolResult(success=False, error=f"Unsupported model {model!r}.")

        output_path, output_error = require_optional_project_sidecar_path(
            inputs, "output_path", self.name, artifact_label="vision analysis"
        )
        if output_error:
            return output_error

        content: list[dict[str, Any]] = [{"type": "text", "text": inputs["prompt"]}]

        image_ref = self._resolve_image(inputs)
        if isinstance(image_ref, ToolResult):
            return image_ref
        if image_ref:
            content.append({"type": "image_url", "image_url": {"url": image_ref}})

        video_url = inputs.get("video_url")
        if video_url:
            content.append({"type": "video_url", "video_url": {"url": video_url}})

        if len(content) == 1:
            return ToolResult(
                success=False,
                error="qwen_vl requires image_url, image_path, or video_url in addition to prompt.",
            )

        api_key = self._get_api_key()
        if not api_key:
            return ToolResult(success=False, error="DASHSCOPE_API_KEY not set. " + self.install_instructions)

        import requests

        start = time.time()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": inputs.get("max_tokens", 2048),
        }
        try:
            resp = requests.post(
                _CHAT_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return ToolResult(success=False, error=f"Qwen-VL request failed: {exc}")

        text = self._extract_text(data)
        if text is None:
            return ToolResult(success=False, error=f"No analysis in response: {data}")

        out_str: str | None = None
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")
            out_str = portable_output_path(output_path)

        usage = data.get("usage", {})
        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": model,
                "model_name": _MODELS[model]["name"],
                "text": text,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "output": out_str,
                "output_path": out_str,
            },
            artifacts=[out_str] if out_str else [],
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            model=model,
        )

    @staticmethod
    def _resolve_image(inputs: dict[str, Any]) -> str | None | ToolResult:
        """Return a URL/data-URL for the image, or None if no image input."""
        if inputs.get("image_url"):
            return inputs["image_url"]
        path_str = inputs.get("image_path")
        if not path_str:
            return None
        path = Path(path_str)
        if not path.exists():
            return ToolResult(success=False, error=f"Image not found: {path_str}")
        suffix = path.suffix.lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(
            suffix, "image/jpeg"
        )
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str | None:
        try:
            content = data["choices"][0]["message"]["content"]
            # Some providers return a list of content blocks; normalize to text.
            if isinstance(content, list):
                text = "".join(b.get("text", "") for b in content if isinstance(b, dict))
            else:
                text = content
            if not isinstance(text, str) or not text.strip():
                return None
            return text
        except (KeyError, IndexError, TypeError):
            return None
