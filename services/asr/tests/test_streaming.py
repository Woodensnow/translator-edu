"""Windowing logic with a fake transcriber."""

from __future__ import annotations

import numpy as np
import pytest

from asr.audio import AudioFormatError, pcm16_to_float
from asr.segment import Segment
from asr.streaming import LiveSession
from support import sine_pcm16


def _session(transcriber, **kwargs) -> LiveSession:
    defaults = dict(
        prompt_for=lambda previous: f"TERMS {previous}".strip(),
        hotwords="backpropagation, gradient descent",
        window_s=4.0,
        partial_s=0.0,
    )
    defaults.update(kwargs)
    return LiveSession(transcriber, **defaults)


def test_windows_are_timestamped_and_carry_previous_text():
    prompts = []

    def transcriber(samples, sample_rate, initial_prompt, hotwords):
        prompts.append((initial_prompt, hotwords, len(samples) / sample_rate))
        return [Segment(0.1, 0.9, "gradient descent")]

    session = _session(transcriber, window_s=1.0)
    first = session.add_pcm16(sine_pcm16(1.0))
    second = session.add_pcm16(sine_pcm16(1.0))
    assert first[0].type == "segment" and first[0].final is True
    assert first[0].id == "seg-1"
    assert second[0].id == "seg-2"
    assert first[0].to_json()["id"] == "seg-1"
    assert first[0].start == pytest.approx(0.1)
    assert second[0].start == pytest.approx(1.1)
    assert prompts[0][0] == "TERMS"
    assert "gradient descent" in prompts[1][0]
    assert prompts[0][1].startswith("backpropagation")


def test_partial_then_final_keeps_the_same_origin():
    calls = []

    def transcriber(samples, sample_rate, initial_prompt, hotwords):
        calls.append(len(samples))
        duration = len(samples) / sample_rate
        return [Segment(0.0, duration, "activation function")]

    session = _session(transcriber, window_s=4.0, partial_s=2.0)
    partial = session.add_samples(pcm16_to_float(sine_pcm16(2.5)), 16000)
    assert [event.type for event in partial] == ["partial"]
    assert partial[0].final is False
    final = session.add_samples(pcm16_to_float(sine_pcm16(1.5)), 16000)
    assert final[-1].type == "segment" and final[-1].final is True
    assert final[-1].start == pytest.approx(0.0)
    assert len(calls) == 2


def test_finish_flushes_a_tail_and_drops_a_blip():
    calls = []

    def transcriber(samples, sample_rate, initial_prompt, hotwords):
        calls.append(len(samples))
        return [Segment(0.0, len(samples) / sample_rate, "backpropagation")]

    session = _session(transcriber, window_s=4.0)
    assert session.add_pcm16(sine_pcm16(1.0)) == []
    flushed = session.finish()
    assert flushed[0].text == "backpropagation"
    assert len(calls) == 1

    short = _session(transcriber, window_s=4.0)
    short.add_pcm16(sine_pcm16(0.1))
    assert short.finish() == []


def test_silence_does_not_call_the_model():
    def transcriber(*args, **kwargs):
        raise AssertionError("silence should not be transcribed")

    session = _session(transcriber, window_s=4.0, partial_s=2.0)
    assert session.add_samples(np.zeros(int(2.5 * 16000), dtype=np.float32), 16000) == []
    assert session.add_samples(np.zeros(int(1.5 * 16000), dtype=np.float32), 16000) == []
    assert session.finish() == []


def test_lagging_status_when_slower_than_realtime():
    ticks = iter([0.0, 5.0])

    def transcriber(samples, sample_rate, initial_prompt, hotwords):
        return [Segment(0.0, 1.0, "backpropagation")]

    session = _session(transcriber, window_s=4.0, clock=lambda: next(ticks))
    events = session.add_pcm16(sine_pcm16(4.0))
    assert events[0].type == "status"
    assert events[0].lagging is True
    assert events[0].to_json()["rtf"] == pytest.approx(1.25)
    assert events[1].final is True


def test_long_buffer_is_split_and_48k_remainder_is_kept():
    calls = []

    def transcriber(samples, sample_rate, initial_prompt, hotwords):
        calls.append(len(samples))
        return [Segment(0.0, 0.5, "gradient descent")]

    session = _session(transcriber, window_s=4.0)
    session.add_pcm16(sine_pcm16(10.0))
    assert len(calls) == 2
    session.finish()
    assert len(calls) == 3

    resampled = LiveSession(
        lambda *args, **kwargs: [],
        prompt_for=lambda previous: "",
        hotwords=None,
        input_sample_rate=48000,
        window_s=30.0,
    )
    resampled.add_samples(np.ones(100, dtype=np.float32), 48000)
    resampled.add_samples(np.ones(48000 - 100, dtype=np.float32), 48000)
    assert resampled.buffer.size == 16000


def test_odd_chunk_is_rejected():
    session = _session(lambda *args, **kwargs: [])
    with pytest.raises(AudioFormatError):
        session.add_pcm16(b"\x00\x01\x02")
