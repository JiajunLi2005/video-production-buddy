"""Transcription tool wrapping faster-whisper / WhisperX.

Provides speech-to-text with word-level timestamps and optional speaker
diarization. Falls back gracefully when GPU or diarization dependencies
are not available.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ResumeSupport,
    ToolResult,
    ToolStability,
    ToolStatus,
    ToolTier,
)
from tools.output_paths import (
    portable_output_path,
    require_explicit_project_sidecar_destination,
)


MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]


class Transcriber(BaseTool):
    name = "transcriber"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "analysis"
    provider = "whisperx"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC

    dependencies = ["python:faster_whisper"]
    install_instructions = (
        "pip install faster-whisper  # CPU mode\n"
        "pip install faster-whisper[gpu]  # GPU mode (requires CUDA)\n"
        "pip install whisperx  # For diarization support"
    )
    agent_skills = ["speech-to-text"]

    capabilities = [
        "transcribe",
        "word_timestamps",
        "diarization",
        "language_detection",
    ]
    best_for = [
        "Local speech transcription with segment and word timing",
        "Preparing captions, subtitles, and transcript-driven edits",
    ]

    input_schema = {
        "type": "object",
        "required": ["input_path", "output_dir"],
        "properties": {
            "input_path": {"type": "string", "description": "Path to audio or video file"},
            "model_size": {
                "type": "string",
                "enum": MODEL_SIZES,
                "default": "base",
            },
            "language": {"type": "string", "description": "ISO 639-1 language code, or null for auto-detect"},
            "diarize": {"type": "boolean", "default": False},
            "output_dir": {
                "type": "string",
                "description": (
                    "Project-scoped directory for transcript files under "
                    "projects/<project-name>/artifacts/..., "
                    "projects/<project-name>/assets/..., or "
                    "projects/<project-name>/renders/..."
                ),
            },
        },
    }

    output_schema = {
        "type": "object",
        "required": [
            "output_dir",
            "transcript_path",
            "segments",
            "word_timestamps",
            "language",
            "duration_seconds",
            "model_size",
            "device",
        ],
        "properties": {
            "output_dir": {"type": "string"},
            "transcript_path": {"type": "string"},
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "start", "end", "text"],
                    "properties": {
                        "id": {"type": "integer"},
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                        "text": {"type": "string"},
                        "words": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["word", "start", "end", "probability"],
                                "properties": {
                                    "word": {"type": "string"},
                                    "start": {"type": "number"},
                                    "end": {"type": "number"},
                                    "probability": {"type": "number"},
                                },
                            },
                        },
                    },
                },
            },
            "word_timestamps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["word", "start", "end", "probability"],
                    "properties": {
                        "word": {"type": "string"},
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                        "probability": {"type": "number"},
                    },
                },
            },
            "language": {"type": "string"},
            "duration_seconds": {"type": "number"},
            "model_size": {"type": "string"},
            "device": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=2,
        ram_mb=2048,
        vram_mb=0,  # CPU by default; GPU optional
        disk_mb=500,
        network_required=False,
    )

    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["MemoryError"])
    resume_support = ResumeSupport.FROM_START
    idempotency_key_fields = ["input_path", "output_dir", "model_size", "language", "diarize"]
    side_effects = ["writes transcript JSON to output_dir"]
    fallback = None
    user_visible_verification = [
        "Check transcript text against source audio",
        "Verify word timestamps align with speech",
    ]

    def get_status(self) -> ToolStatus:
        try:
            import faster_whisper  # noqa: F401
            return ToolStatus.AVAILABLE
        except ImportError:
            return ToolStatus.UNAVAILABLE

    def _has_diarization(self) -> bool:
        try:
            import whisperx  # noqa: F401
            return True
        except ImportError:
            return False

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        """Rough estimate: ~0.5x real-time on CPU for 'base' model."""
        return 60.0  # conservative default

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        input_path = Path(inputs["input_path"])
        model_size = inputs.get("model_size", "base")
        language = inputs.get("language")
        diarize = inputs.get("diarize", False)
        if model_size not in MODEL_SIZES:
            return ToolResult(success=False, error=f"Unknown model_size: {model_size}")

        if not input_path.exists():
            return ToolResult(success=False, error=f"Input file not found: {input_path}")

        output_dir, output_error = require_explicit_project_sidecar_destination(
            inputs,
            "output_dir",
            self.name,
            artifact_label="transcript files",
        )
        if output_error:
            return output_error
        assert output_dir is not None

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            return ToolResult(
                success=False,
                error="faster-whisper is not installed. Run: pip install faster-whisper",
            )

        start = time.time()

        # Load model (CPU by default, CUDA if available)
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
        except ImportError:
            device = "cpu"
            compute_type = "int8"

        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        # Transcribe
        segments_iter, info = model.transcribe(
            str(input_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,
        )

        segments = []
        word_timestamps = []

        for seg in segments_iter:
            seg_data = {
                "id": seg.id,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
            }

            if seg.words:
                words = []
                for w in seg.words:
                    word_entry = {
                        "word": w.word,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 3),
                    }
                    words.append(word_entry)
                    word_timestamps.append(word_entry)
                seg_data["words"] = words

            segments.append(seg_data)

        detected_language = language or info.language
        duration = info.duration

        # Optional diarization pass
        if diarize and self._has_diarization():
            segments = self._apply_diarization(
                str(input_path), segments, detected_language
            )

        elapsed = time.time() - start
        output_path = output_dir / f"{input_path.stem}_transcript.json"

        result_data = {
            "output_dir": portable_output_path(output_dir),
            "transcript_path": portable_output_path(output_path),
            "segments": segments,
            "word_timestamps": word_timestamps,
            "language": detected_language,
            "duration_seconds": round(duration, 3),
            "model_size": model_size,
            "device": device,
        }

        # Write transcript JSON
        try:
            serialized = json.dumps(result_data, indent=2, allow_nan=False)
        except (TypeError, ValueError) as exc:
            return ToolResult(
                success=False,
                error=f"Transcript result must be strict JSON serializable: {exc}",
            )
        output_path.write_text(serialized, encoding="utf-8")

        return ToolResult(
            success=True,
            data=result_data,
            artifacts=[portable_output_path(output_path)],
            duration_seconds=round(elapsed, 2),
        )

    def _apply_diarization(
        self,
        audio_path: str,
        segments: list[dict],
        language: str,
    ) -> list[dict]:
        """Apply WhisperX diarization to assign speaker labels."""
        try:
            import whisperx

            # Load audio for alignment
            audio = whisperx.load_audio(audio_path)

            # Align segments with word timestamps
            align_model, align_metadata = whisperx.load_align_model(
                language_code=language, device="cpu"
            )
            aligned = whisperx.align(
                segments, align_model, align_metadata, audio, device="cpu"
            )

            # Diarize
            import os
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                # Can't diarize without HuggingFace token for pyannote
                return segments

            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token, device="cpu"
            )
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, aligned)

            return result.get("segments", segments)
        except Exception:
            # Diarization is best-effort; return original segments on failure
            return segments
