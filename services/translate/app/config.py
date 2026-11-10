"""Runtime settings for the translation service.

The process is sized for one 2-core CPU server: CTranslate2 on CPU, two intra
threads, one request at a time. There is no GPU setting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

MODEL_NAME = "Helsinki-NLP/opus-mt-en-zh"


def default_glossary_dir() -> Path:
    """Find the shared glossary directory in a checkout or in the image."""
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / "glossary"
        if candidate.is_dir():
            return candidate
    container = Path("/glossary")
    if container.is_dir():
        return container
    return container


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class Settings:
    backend: str = "ctranslate2"
    model_dir: Path = Path("models/opus-mt-en-zh-ct2")
    glossary_dir: Path = field(default_factory=default_glossary_dir)
    default_glossary: str = "deep-learning"
    inter_threads: int = 1
    intra_threads: int = 2
    beam_size: int = 1
    compute_type: str = "int8"
    max_unit_chars: int = 480
    max_chars: int = 20_000
    max_segments: int = 200
    model_name: str = MODEL_NAME

    def __post_init__(self) -> None:
        if self.backend not in {"mock", "ctranslate2"}:
            raise ValueError("TRANSLATE_BACKEND must be mock or ctranslate2")
        if self.inter_threads < 1 or self.intra_threads < 1:
            raise ValueError("thread counts must be at least 1")
        if self.beam_size < 1:
            raise ValueError("beam size must be at least 1")
        if self.max_unit_chars < 32:
            raise ValueError("TRANSLATE_MAX_UNIT_CHARS must be at least 32")
        if self.max_chars < 1 or self.max_segments < 1:
            raise ValueError("request limits must be positive")
        if not self.default_glossary.strip():
            raise ValueError("default glossary name is empty")
        if not self.compute_type.strip():
            raise ValueError("compute type is empty")
        if not self.model_name.strip():
            raise ValueError("model name is empty")

    @classmethod
    def from_env(cls) -> Settings:
        glossary_env = os.environ.get("TRANSLATE_GLOSSARY_DIR", "").strip()
        model_env = os.environ.get("TRANSLATE_MODEL_DIR", "").strip()
        backend = os.environ.get("TRANSLATE_BACKEND", "ctranslate2").strip().lower()
        return cls(
            backend=backend or "ctranslate2",
            model_dir=Path(model_env) if model_env else Path("models/opus-mt-en-zh-ct2"),
            glossary_dir=Path(glossary_env) if glossary_env else default_glossary_dir(),
            default_glossary=os.environ.get("TRANSLATE_DEFAULT_GLOSSARY", "deep-learning").strip()
            or "deep-learning",
            inter_threads=_int_env("TRANSLATE_INTER_THREADS", 1),
            intra_threads=_int_env("TRANSLATE_INTRA_THREADS", 2),
            beam_size=_int_env("TRANSLATE_BEAM_SIZE", 1),
            compute_type=os.environ.get("TRANSLATE_COMPUTE_TYPE", "int8").strip() or "int8",
            max_unit_chars=_int_env("TRANSLATE_MAX_UNIT_CHARS", 480),
            max_chars=_int_env("TRANSLATE_MAX_CHARS", 20_000),
            max_segments=_int_env("TRANSLATE_MAX_SEGMENTS", 200),
            model_name=os.environ.get("TRANSLATE_MODEL_NAME", MODEL_NAME).strip() or MODEL_NAME,
        )
