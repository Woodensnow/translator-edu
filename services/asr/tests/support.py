"""Shared audio helpers for ASR tests. Nothing here touches a model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from asr.audio import wav_bytes_from_pcm16
from asr.segment import Segment

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "deep-learning.csv"


def sine_pcm16(seconds: float, sample_rate: int = 16000, freq: float = 220.0, amp: float = 0.2) -> bytes:
    count = int(seconds * sample_rate)
    timeline = np.arange(count, dtype=np.float32) / sample_rate
    samples = (amp * np.sin(2 * np.pi * freq * timeline) * 32767).astype("<i2")
    return samples.tobytes()


def sine_wav(seconds: float = 0.5) -> bytes:
    return wav_bytes_from_pcm16(sine_pcm16(seconds))


class FakeEngine:
    """Stand-in for WhisperEngine. Records calls and returns one segment."""

    def __init__(self, text: str = "backpropagation uses gradient descent") -> None:
        self.text = text
        self.calls: list[dict] = []

    @property
    def loaded(self) -> bool:
        return False

    def transcribe(self, samples, sample_rate, initial_prompt, hotwords=None, vad=False):
        duration = float(len(samples)) / float(sample_rate)
        self.calls.append(
            {
                "samples": int(len(samples)),
                "sample_rate": int(sample_rate),
                "initial_prompt": initial_prompt,
                "hotwords": hotwords,
                "vad": vad,
                "duration": duration,
            }
        )
        return [Segment(0.1, max(0.2, duration - 0.1), self.text)]


class BoomEngine:
    def transcribe(self, *args, **kwargs):
        raise RuntimeError("weights missing")
