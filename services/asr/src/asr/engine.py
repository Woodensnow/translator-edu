"""faster-whisper wrapper.

The CTranslate2 model is loaded on the first transcription, not at import and
not at process start, so tests and ``/health`` never download weights. Run
``services/asr/scripts/download_model.py`` on the host before a lecture if the
machine cannot reach Hugging Face during class.

``cpu_threads`` must stay at the core count. faster-whisper otherwise defaults
to 4 threads, which oversubscribes a 2-core server. ``beam_size`` is 1 so a
live window finishes inside a few seconds.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from asr.audio import TARGET_SR, resample_to_16k
from asr.segment import Segment
from asr.settings import Settings

logger = logging.getLogger(__name__)


class WhisperEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()
        self._hotwords_supported = True

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from faster_whisper import WhisperModel

            kwargs = {
                "device": "cpu",
                "compute_type": self.settings.compute_type,
                "cpu_threads": self.settings.cpu_threads,
                "num_workers": 1,
                "local_files_only": self.settings.local_files_only,
            }
            if self.settings.model_dir:
                kwargs["download_root"] = self.settings.model_dir
            logger.info(
                "Loading Whisper model %s (%s, %s threads)",
                self.settings.model,
                self.settings.compute_type,
                self.settings.cpu_threads,
            )
            self._model = WhisperModel(self.settings.model, **kwargs)

    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        initial_prompt: str,
        hotwords: str | None = None,
        vad: bool = False,
    ) -> list[Segment]:
        self.ensure_loaded()
        audio = resample_to_16k(samples, sample_rate)
        if audio.size == 0:
            return []
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        kwargs: dict = {
            "language": "en",
            "task": "transcribe",
            "beam_size": 1,
            "best_of": 1,
            "temperature": 0.0,
            "condition_on_previous_text": True,
            "vad_filter": bool(vad),
            "without_timestamps": False,
        }
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        if hotwords and self._hotwords_supported:
            kwargs["hotwords"] = hotwords

        with self._lock:
            try:
                raw = self._generate(audio, kwargs)
            except Exception:
                if not kwargs.get("vad_filter"):
                    raise
                logger.warning("Batch VAD failed; retrying without it", exc_info=True)
                kwargs["vad_filter"] = False
                raw = self._generate(audio, kwargs)

        segments: list[Segment] = []
        for item in raw:
            text = " ".join((getattr(item, "text", "") or "").split())
            if not text:
                continue
            start = float(getattr(item, "start", 0.0) or 0.0)
            end = float(getattr(item, "end", start) or start)
            segments.append(Segment(start=max(0.0, start), end=max(start, end), text=text))
        return segments

    def _generate(self, audio: np.ndarray, kwargs: dict) -> list:
        assert self._model is not None
        try:
            segments, _info = self._model.transcribe(audio, **kwargs)
            return list(segments)
        except TypeError:
            if "hotwords" not in kwargs or not self._hotwords_supported:
                raise
            self._hotwords_supported = False
            logger.warning("This faster-whisper build has no hotwords argument; using the prompt only")
            reduced = dict(kwargs)
            reduced.pop("hotwords", None)
            segments, _info = self._model.transcribe(audio, **reduced)
            return list(segments)
