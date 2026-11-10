"""Audio helpers that do not need Whisper."""

from __future__ import annotations

import shutil
import subprocess
import wave
from io import BytesIO

import numpy as np
import pytest

from asr.audio import (
    TARGET_SR,
    AudioFormatError,
    decode_audio_bytes,
    decode_wav_bytes,
    pcm16_to_float,
    resample_to_16k,
    wav_bytes_from_pcm16,
)
from support import sine_pcm16


def test_pcm16_roundtrip_and_wav():
    pcm = sine_pcm16(0.5)
    floats = pcm16_to_float(pcm)
    assert floats.shape == (8000,)
    assert floats.dtype == np.float32
    decoded, sample_rate = decode_wav_bytes(wav_bytes_from_pcm16(pcm))
    assert sample_rate == TARGET_SR
    assert decoded.shape == (8000,)
    assert np.allclose(decoded, floats, atol=1e-4)


def test_stereo_wav_mixes_to_mono():
    left = sine_pcm16(0.2, freq=220)
    right = sine_pcm16(0.2, freq=440)
    left_i = np.frombuffer(left, dtype="<i2")
    right_i = np.frombuffer(right, dtype="<i2")
    stereo = np.empty(left_i.size * 2, dtype="<i2")
    stereo[0::2] = left_i
    stereo[1::2] = right_i
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(stereo.tobytes())
    decoded, sample_rate = decode_audio_bytes(buffer.getvalue(), "lecture.wav")
    assert sample_rate == 16000
    assert decoded.shape == left_i.shape


def test_resample_integer_and_linear():
    audio = np.ones(48000, dtype=np.float32)
    down = resample_to_16k(audio, 48000)
    assert down.shape == (16000,)
    assert np.allclose(down, 1.0)
    odd = resample_to_16k(np.zeros(44100, dtype=np.float32), 44100)
    assert abs(odd.size - 16000) <= 1


def test_odd_pcm_and_unknown_bytes_raise():
    with pytest.raises(AudioFormatError):
        pcm16_to_float(b"\x00")
    with pytest.raises(AudioFormatError):
        decode_audio_bytes(b"", "empty.wav")
    with pytest.raises(AudioFormatError):
        decode_audio_bytes(b"not-audio", "notes.txt")


def test_mp3_upload_decodes_when_pyav_is_installed(tmp_path):
    pytest.importorskip("av")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    wav_path = tmp_path / "tone.wav"
    mp3_path = tmp_path / "tone.mp3"
    wav_path.write_bytes(wav_bytes_from_pcm16(sine_pcm16(0.4)))
    subprocess.check_call(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path), str(mp3_path)],
        stdout=subprocess.DEVNULL,
    )
    samples, sample_rate = decode_audio_bytes(mp3_path.read_bytes(), "tone.mp3")
    assert sample_rate == TARGET_SR
    assert samples.size > 1000
