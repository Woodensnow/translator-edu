"""Turn a stream of PCM into timestamped English segments.

Audio is cut into fixed windows (default 4 seconds). Each window is one
``transcribe`` call. That keeps a one-hour lecture from being reprocessed from
the start, and it bounds latency to about one window plus the model time.

Partials are off by default (``ASR_PARTIAL_SECONDS=0``) so a 2-core host does
not transcribe the same audio twice. Set the variable to something like ``2``
when the measured real-time factor has spare room.

Silence below ``silence_rms`` does not call the model. Windows that run slower
than real time emit a ``status`` event with ``lagging: true`` so the browser
can keep a recording and upload it afterwards.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from asr.audio import TARGET_SR, pcm16_to_float, resample_to_16k
from asr.segment import Segment

logger = logging.getLogger(__name__)


class Transcriber(Protocol):
    def __call__(
        self,
        samples: np.ndarray,
        sample_rate: int,
        initial_prompt: str,
        hotwords: str | None,
    ) -> list[Segment]:
        """Transcribe one window. Timestamps are relative to ``samples``."""


PromptFor = Callable[[str], str]
Clock = Callable[[], float]


@dataclass(frozen=True)
class TranscriptEvent:
    type: str
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    final: bool = False
    lagging: bool = False
    rtf: float | None = None
    id: str = ""

    def to_json(self) -> dict:
        if self.type == "status":
            body: dict = {"type": "status", "lagging": self.lagging}
            if self.rtf is not None:
                body["rtf"] = round(self.rtf, 3)
            return body
        payload = {
            "type": self.type,
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "text": self.text,
            "final": self.final,
        }
        if self.id:
            payload["id"] = self.id
        return payload


class LiveSession:
    """One lecturer connection. Not safe to call from two threads at once."""

    def __init__(
        self,
        transcriber: Transcriber,
        *,
        prompt_for: PromptFor,
        hotwords: str | None,
        input_sample_rate: int = TARGET_SR,
        window_s: float = 4.0,
        partial_s: float = 0.0,
        silence_rms: float = 0.008,
        min_final_s: float = 0.3,
        clock: Clock | None = None,
        id_prefix: str = "",
    ) -> None:
        if input_sample_rate <= 0:
            raise ValueError("input_sample_rate must be positive.")
        self.transcriber = transcriber
        self.prompt_for = prompt_for
        self.hotwords = hotwords
        self.input_sample_rate = int(input_sample_rate)
        self.window_s = float(window_s)
        self.partial_s = float(partial_s)
        self.silence_rms = float(silence_rms)
        self.min_final_s = float(min_final_s)
        self.clock = clock or time.perf_counter
        self.buffer = np.zeros(0, dtype=np.float32)
        self.origin = 0.0
        self.partial_sent = False
        self.last_text = ""
        self._rate_remainder = np.zeros(0, dtype=np.float32)
        self._last_rate = self.input_sample_rate
        self._seq = 0
        self.id_prefix = str(id_prefix or "")

    def add_pcm16(self, data: bytes) -> list[TranscriptEvent]:
        samples = pcm16_to_float(data)
        return self.add_samples(samples, self.input_sample_rate)

    def add_samples(self, samples: np.ndarray, sample_rate: int) -> list[TranscriptEvent]:
        converted = self._ingest(samples, sample_rate)
        if converted.size == 0:
            return []
        self.buffer = np.concatenate([self.buffer, converted])
        return self._drain(final=False)

    def finish(self) -> list[TranscriptEvent]:
        return self._drain(final=True)

    def _ingest(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return audio
        if sample_rate != self._last_rate:
            self._rate_remainder = np.zeros(0, dtype=np.float32)
            self._last_rate = sample_rate
        if sample_rate == TARGET_SR or sample_rate % TARGET_SR != 0:
            return resample_to_16k(audio, sample_rate)

        factor = sample_rate // TARGET_SR
        pending = np.concatenate([self._rate_remainder, audio]) if self._rate_remainder.size else audio
        usable = (pending.size // factor) * factor
        self._rate_remainder = pending[usable:]
        if usable == 0:
            return np.zeros(0, dtype=np.float32)
        return resample_to_16k(pending[:usable], sample_rate)

    def _drain(self, final: bool) -> list[TranscriptEvent]:
        events: list[TranscriptEvent] = []
        window_n = max(1, int(round(self.window_s * TARGET_SR)))
        partial_n = 0
        if 0 < self.partial_s < self.window_s:
            partial_n = int(round(self.partial_s * TARGET_SR))
        min_final_n = int(round(self.min_final_s * TARGET_SR))

        while True:
            count = int(self.buffer.size)
            if count >= window_n:
                chunk = self.buffer[:window_n]
                self.buffer = self.buffer[window_n:]
                events.extend(self._recognize(chunk, final_segment=True))
                self.origin += window_n / TARGET_SR
                self.partial_sent = False
                continue
            if partial_n and not self.partial_sent and count >= partial_n and not final:
                events.extend(self._recognize(self.buffer, final_segment=False))
                self.partial_sent = True
                break
            if final and count > 0:
                chunk = self.buffer
                self.buffer = np.zeros(0, dtype=np.float32)
                if count >= min_final_n:
                    events.extend(self._recognize(chunk, final_segment=True))
                self.origin += count / TARGET_SR
                self.partial_sent = False
            break
        return events

    def _recognize(self, chunk: np.ndarray, final_segment: bool) -> list[TranscriptEvent]:
        chunk = np.ascontiguousarray(chunk, dtype=np.float32)
        duration = chunk.size / TARGET_SR
        if duration <= 0 or _rms(chunk) < self.silence_rms:
            return []

        prompt = self.prompt_for(self.last_text)
        started = float(self.clock())
        segments = self.transcriber(chunk, TARGET_SR, prompt, self._hotwords_now())
        elapsed = float(self.clock()) - started
        events: list[TranscriptEvent] = []
        rtf = elapsed / duration if duration else 0.0
        if rtf > 1.0:
            logger.warning("transcription slower than real time (rtf=%.2f)", rtf)
            events.append(TranscriptEvent(type="status", lagging=True, rtf=rtf))

        cleaned: list[tuple[Segment, str]] = []
        for segment in segments:
            text = " ".join((segment.text or "").split())
            if text:
                cleaned.append((segment, text))
        if not cleaned:
            return events

        if not final_segment:
            text = " ".join(text for _, text in cleaned)
            start = self.origin + max(0.0, float(cleaned[0][0].start))
            events.append(
                TranscriptEvent(
                    type="partial",
                    start=start,
                    end=self.origin + duration,
                    text=text,
                    final=False,
                )
            )
            return events

        pieces: list[str] = []
        for segment, text in cleaned:
            start = self.origin + max(0.0, float(segment.start))
            end_rel = max(float(segment.end), float(segment.start))
            end = self.origin + min(duration, end_rel)
            events.append(
                TranscriptEvent(
                    type="segment",
                    id=self._next_id(),
                    start=start,
                    end=end,
                    text=text,
                    final=True,
                )
            )
            pieces.append(text)
        if pieces:
            self.last_text = pieces[-1]
        return events


    def _hotwords_now(self) -> str | None:
        source = self.hotwords
        if callable(source):
            return source()
        return source

    def _next_id(self) -> str:
        self._seq += 1
        stem = f"seg-{self._seq}"
        if self.id_prefix:
            return f"{self.id_prefix}-{stem}"
        return stem


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
