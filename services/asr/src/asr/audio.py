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


class AudioTooLong(AudioFormatError):
    """The recording is longer than ASR_MAX_AUDIO_SECONDS. Never truncated."""

    def __init__(self, max_seconds: float) -> None:
        self.max_seconds = float(max_seconds)
        super().__init__(f"录音太长，请分段上传（上限 {int(self.max_seconds)} 秒）")


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


def media_duration_seconds(path: str) -> float | None:
    """Best-effort duration. None when the container does not say."""

    if _looks_like_wav(path):
        with wave.open(path, "rb") as handle:
            rate = handle.getframerate()
            if rate <= 0:
                return None
            return handle.getnframes() / float(rate)
    try:
        import av
    except ImportError:
        return None
    try:
        container = av.open(path)
    except Exception:
        return None
    try:
        if container.duration is not None and container.duration > 0:
            return float(container.duration) / 1_000_000.0
        stream = next((item for item in container.streams if item.type == "audio"), None)
        if stream is None or not stream.duration or not stream.time_base:
            return None
        return float(stream.duration * stream.time_base)
    finally:
        container.close()


def iter_pcm_windows(path: str, *, window_s: float, max_seconds: float):
    """Yield ``(float32 16 kHz mono, offset_seconds, window_seconds)``.

    One window is in memory at a time. If the recording passes ``max_seconds``
    this raises ``AudioTooLong`` and does not yield the overflowing audio.
    Callers must not treat earlier windows as a successful short transcript.
    """

    if max_seconds <= 0:
        raise AudioTooLong(max_seconds)
    known = media_duration_seconds(path)
    if known is not None and known > max_seconds + 1e-3:
        raise AudioTooLong(max_seconds)
    emitter = _WindowEmitter(window_s, max_seconds)
    if _looks_like_wav(path):
        blocks = _iter_wav_blocks(path)
    else:
        blocks = _iter_pyav_blocks(path)
    for block in blocks:
        yield from emitter.push(block)
    yield from emitter.finish()


class _WindowEmitter:
    def __init__(self, window_s: float, max_seconds: float) -> None:
        self.window_n = max(1, int(round(window_s * TARGET_SR)))
        self.max_n = max(1, int(round(max_seconds * TARGET_SR)))
        self.max_seconds = max_seconds
        self.buffer = np.zeros(0, dtype=np.float32)
        self.total = 0
        self.offset = 0.0

    def push(self, samples: np.ndarray):
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return []
        if self.total + int(audio.size) > self.max_n:
            raise AudioTooLong(self.max_seconds)
        self.total += int(audio.size)
        self.buffer = np.concatenate([self.buffer, audio])
        return self._drain(final=False)

    def finish(self):
        return self._drain(final=True)

    def _drain(self, final: bool):
        ready = []
        while self.buffer.size >= self.window_n:
            chunk = np.ascontiguousarray(self.buffer[: self.window_n])
            self.buffer = self.buffer[self.window_n :]
            duration = self.window_n / TARGET_SR
            ready.append((chunk, self.offset, duration))
            self.offset += duration
        if final and self.buffer.size:
            chunk = np.ascontiguousarray(self.buffer)
            self.buffer = np.zeros(0, dtype=np.float32)
            duration = chunk.size / TARGET_SR
            ready.append((chunk, self.offset, duration))
            self.offset += duration
        return ready


def _looks_like_wav(path: str) -> bool:
    if str(path).lower().endswith(".wav"):
        return True
    try:
        with open(path, "rb") as handle:
            return handle.read(4) == b"RIFF"
    except OSError:
        return False


def _iter_wav_blocks(path: str):
    with wave.open(path, "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        if rate <= 0:
            raise AudioFormatError("WAV sample rate is invalid.")
        frames_per_block = max(1, rate)  # about one second
        while True:
            raw = handle.readframes(frames_per_block)
            if not raw:
                break
            samples = _frames_to_float(raw, width, channels)
            yield resample_to_16k(samples, rate)


def _frames_to_float(raw: bytes, width: int, channels: int) -> np.ndarray:
    if width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioFormatError(f"Unsupported WAV sample width {width}.")
    if channels > 1:
        usable = (samples.size // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32)


def _iter_pyav_blocks(path: str):
    try:
        import av
    except ImportError as exc:
        raise AudioFormatError(
            "Only WAV uploads can be decoded in this environment. "
            "Install PyAV (it comes with faster-whisper) or upload a WAV file."
        ) from exc
    try:
        container = av.open(path)
    except Exception as exc:
        raise AudioFormatError(f"Could not decode audio ({exc}).") from exc
    try:
        stream = next((item for item in container.streams if item.type == "audio"), None)
        if stream is None:
            raise AudioFormatError("The file has no audio stream.")
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=TARGET_SR)
        for frame in container.decode(audio=0):
            for chunk in _resampler_chunks(resampler.resample(frame)):
                yield chunk.astype(np.float32) / 32768.0
        for chunk in _resampler_chunks(resampler.resample(None)):
            yield chunk.astype(np.float32) / 32768.0
    finally:
        container.close()


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
