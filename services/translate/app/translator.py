"""Translation backends.

``mock`` copies the protected text so glossary tests and local wiring work
without weights. ``ctranslate2`` runs Helsinki-NLP/opus-mt-en-zh on CPU.
PyTorch is not imported here; weights are converted ahead of time.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

from app.config import Settings

logger = logging.getLogger("translate.backend")

_SPECIAL_TOKENS = {"<s>", "</s>", "<unk>", "<pad>"}
_MAX_INPUT_TOKENS = 256
_MAX_DECODING_LENGTH = 256
# Health checks must not reload on every poll. Docker's interval is 30s.
LOAD_RETRY_SECONDS = 30.0


class ModelNotReady(RuntimeError):
    """Weights are missing or the CPU runtime failed to start."""


class MockTranslator:
    """Echo text. Glossary substitution still runs around this backend."""

    name = "mock"

    def health(self) -> tuple[bool, str | None]:
        return True, None

    def prepare(self) -> tuple[bool, str | None]:
        return self.health()

    def translate_many(self, texts: list[str]) -> list[str]:
        return list(texts)


class CTranslate2Translator:
    """int8 CTranslate2 runtime. Device is CPU even if the host has a GPU."""

    name = "ctranslate2"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._next_load_attempt = 0.0
        self._last_load_error: str | None = None
        self._model_dir: Path | None = None
        self._source = None
        self._target = None
        self._translator = None

    def health(self) -> tuple[bool, str | None]:
        """Report readiness without taking the inference lock.

        A failed startup load is retried from here, rate-limited, and only when
        the load lock is free. A caption in progress is not queued behind.
        """
        if self._translator is not None:
            return True, None
        problem = self._availability_error()
        if problem:
            return False, problem
        self._retry_load()
        if self._translator is not None:
            return True, None
        return False, self._last_load_error or "model is not loaded"

    def prepare(self) -> tuple[bool, str | None]:
        try:
            with self._load_lock:
                self._load_if_needed()
        except Exception as exc:
            self._last_load_error = str(exc)
            return False, self._last_load_error
        self._last_load_error = None
        return True, None

    def _retry_load(self) -> None:
        """Try to load once if the retry window has elapsed. Never blocks."""
        if self._translator is not None:
            return
        now = time.monotonic()
        if now < self._next_load_attempt:
            return
        if not self._load_lock.acquire(blocking=False):
            return
        try:
            if self._translator is not None:
                return
            now = time.monotonic()
            if now < self._next_load_attempt:
                return
            self._next_load_attempt = now + LOAD_RETRY_SECONDS
            try:
                self._load_if_needed()
            except Exception as exc:
                self._last_load_error = str(exc)
                logger.warning("model load failed; health will retry later: %s", exc)
            else:
                self._last_load_error = None
        finally:
            self._load_lock.release()

    def _availability_error(self) -> str | None:
        try:
            model_dir = self._resolve_model_dir()
        except ModelNotReady as exc:
            return str(exc)
        missing = [name for name in ("model.bin", "source.spm", "target.spm") if not (model_dir / name).is_file()]
        if missing:
            return (
                f"CTranslate2 model at {model_dir} is missing {', '.join(missing)}. "
                "Convert weights once with: python scripts/download_model.py"
            )
        return None

    def translate_many(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        outputs = [""] * len(texts)
        indexes: list[int] = []
        batch: list[str] = []
        for index, text in enumerate(texts):
            if text.strip():
                indexes.append(index)
                batch.append(text)
            else:
                outputs[index] = text
        if not batch:
            return outputs
        with self._lock:
            try:
                with self._load_lock:
                    self._load_if_needed()
                translated = self._translate_loaded(batch)
            except ModelNotReady:
                raise
            except Exception as exc:
                logger.exception("ctranslate2 translation failed")
                raise ModelNotReady(str(exc)) from exc
        if len(translated) != len(batch):
            raise ModelNotReady("ctranslate2 returned a different number of translations")
        for index, value in zip(indexes, translated):
            outputs[index] = value
        return outputs

    def _candidate_dirs(self) -> list[Path]:
        path = self.settings.model_dir
        if path.is_absolute():
            return [path]
        service_root = Path(__file__).resolve().parents[1]
        return [Path.cwd() / path, service_root / path]

    def _resolve_model_dir(self) -> Path:
        candidates = self._candidate_dirs()
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        looked = ", ".join(str(candidate) for candidate in candidates)
        raise ModelNotReady(
            "CTranslate2 model directory not found. "
            f"Looked for: {looked}. "
            "Convert weights once with: python scripts/download_model.py"
        )

    def _resolve_compute_type(self, model_dir: Path) -> str:
        if os.environ.get("TRANSLATE_COMPUTE_TYPE", "").strip():
            return self.settings.compute_type
        meta = model_dir / "runtime.json"
        if meta.is_file():
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            value = data.get("compute_type") if isinstance(data, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip()
        return self.settings.compute_type

    def _load_if_needed(self) -> None:
        if self._translator is not None:
            return
        model_dir = self._resolve_model_dir()
        required = ("model.bin", "source.spm", "target.spm")
        missing = [name for name in required if not (model_dir / name).is_file()]
        if missing:
            raise ModelNotReady(
                f"CTranslate2 model at {model_dir} is missing {', '.join(missing)}. "
                "Convert weights once with: python scripts/download_model.py"
            )
        try:
            import ctranslate2
            import sentencepiece as spm
        except ImportError as exc:
            raise ModelNotReady(
                "ctranslate2 and sentencepiece are required for the CPU backend. "
                "Install services/translate/requirements.txt."
            ) from exc

        source = spm.SentencePieceProcessor()
        target = spm.SentencePieceProcessor()
        source.Load(str(model_dir / "source.spm"))
        target.Load(str(model_dir / "target.spm"))
        self._encode = _encode_pieces
        self._decode = _decode_pieces
        compute_type = self._resolve_compute_type(model_dir)
        self._source = source
        self._target = target
        self._model_dir = model_dir
        self._translator = ctranslate2.Translator(
            str(model_dir),
            device="cpu",
            inter_threads=self.settings.inter_threads,
            intra_threads=self.settings.intra_threads,
            compute_type=compute_type,
        )
        logger.info("loaded %s (%s) on cpu", model_dir, compute_type)

    def _translate_loaded(self, texts: list[str]) -> list[str]:
        tokenized: list[list[str]] = []
        for text in texts:
            pieces = self._encode(self._source, text)
            if not pieces or pieces[-1] != "</s>":
                pieces.append("</s>")
            if len(pieces) > _MAX_INPUT_TOKENS:
                pieces = pieces[: _MAX_INPUT_TOKENS - 1] + ["</s>"]
            tokenized.append(pieces)
        results = self._translator.translate_batch(
            tokenized,
            beam_size=self.settings.beam_size,
            max_decoding_length=_MAX_DECODING_LENGTH,
        )
        if len(results) != len(tokenized):
            raise ModelNotReady("ctranslate2 returned a different number of translations")
        decoded: list[str] = []
        for result in results:
            hypothesis = result.hypotheses[0] if result.hypotheses else []
            tokens = [token for token in hypothesis if token not in _SPECIAL_TOKENS]
            if not tokens:
                decoded.append("")
            else:
                decoded.append(self._decode(self._target, tokens).strip())
        return decoded


def _encode_pieces(processor: object, text: str) -> list[str]:
    encode = getattr(processor, "encode", None)
    if encode is not None:
        return list(encode(text, out_type=str))
    return list(processor.EncodeAsPieces(text))  # type: ignore[attr-defined]


def _decode_pieces(processor: object, tokens: list[str]) -> str:
    decode = getattr(processor, "decode", None)
    if decode is not None:
        return str(decode(tokens))
    return str(processor.DecodePieces(tokens))  # type: ignore[attr-defined]


def build_translator(settings: Settings) -> MockTranslator | CTranslate2Translator:
    if settings.backend == "mock":
        return MockTranslator()
    if settings.backend == "ctranslate2":
        return CTranslate2Translator(settings)
    raise ValueError("TRANSLATE_BACKEND must be mock or ctranslate2")
