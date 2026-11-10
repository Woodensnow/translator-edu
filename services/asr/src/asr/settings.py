"""Environment configuration for the ASR process.

Each variable is read once at startup so the same image can run on the
transcription host with a glossary path and model cache that differ from a
laptop. See ``docs/asr.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from asr.glossary import DEFAULT_GLOSSARY_PATH, glossary_path_from_env

# English-only int8 is the 2-core default. docs/deploy.md records the measured
# real-time factor and when to switch ASR_MODEL to tiny.en.
DEFAULT_MODEL = "base.en"


@dataclass(frozen=True)
class Settings:
    model: str = DEFAULT_MODEL
    compute_type: str = "int8"
    cpu_threads: int = 2
    model_dir: str | None = None
    local_files_only: bool = False
    glossary_path: Path = Path(DEFAULT_GLOSSARY_PATH)
    window_s: float = 4.0
    partial_s: float = 0.0
    silence_rms: float = 0.008
    prompt_max_chars: int = 500
    hotword_max_chars: int = 400
    cors_origins: tuple[str, ...] = ("*",)
    max_upload_bytes: int = 100 * 1024 * 1024
    max_audio_seconds: float = 7200.0
    batch_window_s: float = 45.0
    batch_vad: bool = True
    host: str = "0.0.0.0"
    port: int = 8001

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("ASR_MODEL must not be empty.")
        if self.cpu_threads < 1:
            raise ValueError("ASR_CPU_THREADS must be >= 1.")
        if not 1 <= self.window_s <= 30:
            raise ValueError("ASR_WINDOW_SECONDS must be between 1 and 30.")
        if self.partial_s < 0:
            raise ValueError("ASR_PARTIAL_SECONDS must be >= 0.")
        if self.prompt_max_chars < 32:
            raise ValueError("ASR_PROMPT_MAX_CHARS must be >= 32.")
        if self.max_upload_bytes < 1:
            raise ValueError("ASR_MAX_UPLOAD_MB must be >= 1.")
        if self.max_audio_seconds < 1:
            raise ValueError("ASR_MAX_AUDIO_SECONDS must be >= 1.")
        if not 1 <= self.batch_window_s <= 120:
            raise ValueError("ASR_BATCH_WINDOW_SECONDS must be between 1 and 120.")
        if not 1 <= self.port <= 65535:
            raise ValueError("ASR_PORT must be a TCP port.")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        source = os.environ if env is None else env
        model_dir = str(source.get("ASR_MODEL_DIR", "")).strip() or None
        origins = _origins(str(source.get("ASR_CORS_ORIGINS", "*")))
        return cls(
            model=str(source.get("ASR_MODEL", DEFAULT_MODEL)).strip() or DEFAULT_MODEL,
            compute_type=str(source.get("ASR_COMPUTE_TYPE", "int8")).strip() or "int8",
            cpu_threads=_int(source, "ASR_CPU_THREADS", 2),
            model_dir=model_dir,
            local_files_only=_bool(source, "ASR_LOCAL_FILES_ONLY", False),
            glossary_path=glossary_path_from_env(source),
            window_s=_float(source, "ASR_WINDOW_SECONDS", 4.0),
            partial_s=_float(source, "ASR_PARTIAL_SECONDS", 0.0),
            silence_rms=_float(source, "ASR_SILENCE_RMS", 0.008),
            prompt_max_chars=_int(source, "ASR_PROMPT_MAX_CHARS", 500),
            hotword_max_chars=_int(source, "ASR_HOTWORD_MAX_CHARS", 400),
            cors_origins=origins,
            max_upload_bytes=_int(source, "ASR_MAX_UPLOAD_MB", 100) * 1024 * 1024,
            max_audio_seconds=_float(source, "ASR_MAX_AUDIO_SECONDS", 7200.0),
            batch_window_s=_float(source, "ASR_BATCH_WINDOW_SECONDS", 45.0),
            batch_vad=_bool(source, "ASR_BATCH_VAD", True),
            host=str(source.get("ASR_HOST", "0.0.0.0")).strip() or "0.0.0.0",
            port=_int(source, "ASR_PORT", 8001),
        )


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}.") from exc


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"{key} must be a number, got {raw!r}.") from exc


def _bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _origins(raw: str) -> tuple[str, ...]:
    if raw.strip() in {"", "*"}:
        return ("*",)
    parts = tuple(part.strip() for part in raw.split(",") if part.strip())
    return parts or ("*",)
