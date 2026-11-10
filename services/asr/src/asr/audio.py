"""PCM and upload decoding.

Streaming clients send 16 kHz mono PCM16 little-endian. Uploads may be WAV
(decoded with the standard library) or another format PyAV can open. PyAV ships
with faster-whisper; the unit tests stay on WAV so CI does not need it.
"""

from __future__ import annotations

import io
import wave

import numpy as np

TARGET_SR = 16000


class AudioFormatError(ValueError):
    """The bytes are not audio this service can decode."""


def pcm16_to_float(data: bytes) -> np.ndarray:
    """Little-endian signed 16-bit PCM to float32 in [-1, 1]."""

    if not data:
        return np.zeros(0, dtype=np.float32)
    if len(data) % 2 != 0:
        raise AudioFormatError("PCM16 chunks must have an even number of bytes.")
    samples = np.frombuffer(data, dtype="<i2").astype(np.float32)
    return samples / 32768.0


def resample_to_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert mono float audio to 16 kHz.

    Integer downsampling (48 kHz, 32 kHz) uses a box filter. Callers that stream
    those rates should keep leftover samples themselves; this function converts
    one contiguous buffer.
    """

    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    if sample_rate == TARGET_SR:
        return np.ascontiguousarray(audio)
    if sample_rate <= 0:
        raise AudioFormatError(f"Invalid sample rate {sample_rate}.")
    if sample_rate % TARGET_SR == 0:
        factor = sample_rate // TARGET_SR
        usable = (audio.size // factor) * factor
        if usable == 0:
            return np.zeros(0, dtype=np.float32)
        grouped = audio[:usable].reshape(-1, factor)
        return np.ascontiguousarray(grouped.mean(axis=1), dtype=np.float32)

    duration = audio.size / float(sample_rate)
    target_len = max(1, int(round(duration * TARGET_SR)))
    src_x = np.linspace(0.0, duration, num=audio.size, endpoint=False)
    dst_x = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def decode_audio_bytes(data: bytes, filename: str = "") -> tuple[np.ndarray, int]:
    """Decode an upload to mono float32 and its sample rate.

    WAV is returned at its native rate. Other containers are resampled to
    16 kHz by PyAV and reported as 16 kHz.
    """

    if not data:
        raise AudioFormatError("The upload was empty.")
    name = (filename or "").lower()
    if name.endswith(".wav") or data[:4] == b"RIFF":
        return decode_wav_bytes(data)
    return decode_with_pyav(data)


def decode_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            return _read_wav(handle)
    except AudioFormatError:
        raise
    except Exception as exc:
        raise AudioFormatError(f"Could not read WAV audio ({exc}).") from exc


def _read_wav(handle: wave.Wave_read) -> tuple[np.ndarray, int]:
    sample_rate = handle.getframerate()
    channels = handle.getnchannels()
    width = handle.getsampwidth()
    frames = handle.readframes(handle.getnframes())
    if sample_rate <= 0:
        raise AudioFormatError("WAV sample rate is invalid.")
    if width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioFormatError(f"Unsupported WAV sample width {width}.")
    if channels > 1:
        usable = (samples.size // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32), int(sample_rate)


def decode_with_pyav(data: bytes) -> tuple[np.ndarray, int]:
    try:
        import av
    except ImportError as exc:
        raise AudioFormatError(
            "Only WAV uploads can be decoded in this environment. "
            "Install PyAV (it comes with faster-whisper) or upload a WAV file."
        ) from exc

    try:
        container = av.open(io.BytesIO(data))
    except Exception as exc:
        raise AudioFormatError(f"Could not decode audio ({exc}).") from exc

    try:
        stream = next((item for item in container.streams if item.type == "audio"), None)
        if stream is None:
            raise AudioFormatError("The file has no audio stream.")
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=TARGET_SR)
        pieces: list[np.ndarray] = []
        for frame in container.decode(audio=0):
            pieces.extend(_resampler_chunks(resampler.resample(frame)))
        pieces.extend(_resampler_chunks(resampler.resample(None)))
    except AudioFormatError:
        raise
    except Exception as exc:
        raise AudioFormatError(f"Could not decode audio ({exc}).") from exc
    finally:
        container.close()

    if not pieces:
        raise AudioFormatError("The audio file had no samples.")
    samples = np.concatenate(pieces).astype(np.float32) / 32768.0
    return np.ascontiguousarray(samples, dtype=np.float32), TARGET_SR


def _resampler_chunks(result: object) -> list[np.ndarray]:
    if result is None:
        return []
    frames = result if isinstance(result, list) else [result]
    chunks: list[np.ndarray] = []
    for frame in frames:
        array = frame.to_ndarray()
        chunks.append(np.asarray(array).reshape(-1).astype(np.int16))
    return chunks


def wav_bytes_from_pcm16(pcm: bytes, sample_rate: int = TARGET_SR) -> bytes:
    """Pack PCM16 mono into a WAV container. Used by tests and local tools."""

    if len(pcm) % 2 != 0:
        raise AudioFormatError("PCM16 chunks must have an even number of bytes.")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()
