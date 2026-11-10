"""HTTP and WebSocket API for lecture transcription.

Batch fallback::

    POST /transcribe
    multipart field ``file``
    -> {"language": "en", "text": "...", "segments": [{"id", "start", "end", "text"}]}

Live audio::

    WebSocket /ws/transcribe
    client text  {"type": "start", "sample_rate": 16000}
    client bytes PCM16 LE mono at that rate (16 kHz preferred)
    client text  {"type": "stop"}
    server       {"type": "ready", ...}
                 {"type": "segment", "id", "start", "end", "text", "final": true}
                 {"type": "partial", ...}          # only if partials are enabled
                 {"type": "status", "lagging": true, "rtf": 1.2}
                 {"type": "done"}

The glossary path is process-wide (``ASR_GLOSSARY_PATH``). One course per ASR
process. Switching courses means pointing the variable at
``glossary/<course>.csv`` and restarting.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import math
import os
import tempfile
import threading
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from asr.audio import TARGET_SR, AudioFormatError, AudioTooLong, iter_pcm_windows, media_duration_seconds
from asr.engine import WhisperEngine
from asr.glossary import GlossaryPrompt
from asr.settings import Settings
from asr.streaming import LiveSession

logger = logging.getLogger(__name__)


class SegmentOut(BaseModel):
    id: str
    start: float
    end: float
    text: str


class TranscriptOut(BaseModel):
    language: str = "en"
    text: str
    segments: list[SegmentOut] = Field(default_factory=list)


def create_app(settings: Settings | None = None, engine: Any | None = None) -> FastAPI:
    """Build the ASGI app. Pass ``engine`` from tests to avoid loading Whisper."""

    _configure_logging()
    resolved = settings or Settings.from_env()
    glossary = GlossaryPrompt(
        resolved.glossary_path,
        max_chars=resolved.prompt_max_chars,
        hotword_max_chars=resolved.hotword_max_chars,
    )
    glossary.refresh()
    logger.info(
        "Glossary %s has %d English terms",
        resolved.glossary_path,
        len(glossary.terms),
    )
    model = engine if engine is not None else WhisperEngine(resolved)
    serial = itertools.count(1)
    uploads: dict[str, _UploadCursor] = {}

    app = FastAPI(title="translator-edu ASR", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = resolved
    app.state.engine = model
    app.state.glossary = glossary

    @app.get("/")
    def root() -> dict:
        return {
            "service": "translator-edu-asr",
            "health": "/health",
            "batch": "POST /transcribe",
            "stream": "WS /ws/transcribe",
        }

    @app.get("/health")
    def health() -> dict:
        glossary.refresh()
        return {
            "status": "ok",
            "model": resolved.model,
            "compute_type": resolved.compute_type,
            "cpu_threads": resolved.cpu_threads,
            "glossary_path": str(resolved.glossary_path),
            "glossary_terms": len(glossary.terms),
            "window_s": resolved.window_s,
            "model_loaded": bool(getattr(model, "loaded", False)),
        }

    @app.post("/transcribe", response_model=TranscriptOut)
    async def transcribe(file: UploadFile = File(...)) -> TranscriptOut:
        path = await _save_upload(file, resolved.max_upload_bytes, file.filename or "")
        prefix = f"b{next(serial)}"
        try:
            segments = await asyncio.to_thread(_transcribe_file, path, resolved, model, glossary, prefix)
        except AudioTooLong as exc:
            raise _too_long(exc) from exc
        except AudioFormatError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Batch transcription failed")
            raise HTTPException(status_code=500, detail="Transcription failed.") from exc
        finally:
            _unlink(path)
        return TranscriptOut(
            language="en",
            text=" ".join(item.text for item in segments),
            segments=segments,
        )

    @app.post("/uploads")
    async def create_upload(request: Request) -> dict:
        """Store one recording on disk and return its duration. No PCM is kept."""

        filename = request.headers.get("x-filename") or "lecture.bin"
        path = await _save_chunks(
            request.stream(),
            resolved.max_upload_bytes,
            filename,
            request.headers.get("content-length"),
        )
        try:
            duration = media_duration_seconds(path)
            if duration is not None and duration > resolved.max_audio_seconds + 1e-3:
                raise AudioTooLong(resolved.max_audio_seconds)
            cursor = _UploadCursor(path, resolved, model, glossary, f"b{next(serial)}", duration)
        except AudioTooLong as exc:
            _unlink(path)
            raise _too_long(exc) from exc
        except Exception:
            _unlink(path)
            raise
        uploads[cursor.upload_id] = cursor
        windows = None
        if duration is not None:
            windows = max(1, int(math.ceil((duration - 1e-6) / resolved.batch_window_s)))
        return {
            "id": cursor.upload_id,
            "duration": duration,
            "window_seconds": resolved.batch_window_s,
            "windows": windows,
            "max_seconds": int(resolved.max_audio_seconds),
        }

    @app.post("/uploads/{upload_id}/next")
    async def next_window(upload_id: str) -> dict:
        cursor = uploads.get(upload_id)
        if cursor is None:
            raise HTTPException(status_code=404, detail="Upload was not found.")
        try:
            return await asyncio.to_thread(cursor.pull)
        except AudioTooLong as exc:
            _discard_upload(uploads, upload_id)
            raise _too_long(exc) from exc

    @app.delete("/uploads/{upload_id}")
    async def delete_upload(upload_id: str) -> dict:
        _discard_upload(uploads, upload_id)
        return {"ok": True}

    @app.websocket("/ws/transcribe")
    async def ws_transcribe(socket: WebSocket) -> None:
        await socket.accept()
        session: LiveSession | None = None
        try:
            while True:
                message = await socket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("text") is not None:
                    session = await _handle_text(
                        socket,
                        session,
                        message["text"],
                        resolved,
                        model,
                        glossary,
                        serial,
                    )
                elif message.get("bytes") is not None:
                    if session is None:
                        await socket.send_json({"type": "error", "message": "Send start before audio."})
                        continue
                    try:
                        events = await asyncio.to_thread(session.add_pcm16, message["bytes"])
                    except AudioFormatError as exc:
                        await socket.send_json({"type": "error", "message": str(exc)})
                        continue
                    except Exception:
                        logger.exception("Live transcription failed")
                        await socket.send_json({"type": "error", "message": "Transcription failed."})
                        continue
                    for event in events:
                        await socket.send_json(event.to_json())
        except WebSocketDisconnect:
            return

    return app


async def _handle_text(
    socket: WebSocket,
    session: LiveSession | None,
    raw: str,
    settings: Settings,
    engine: Any,
    glossary: GlossaryPrompt,
    serial,
) -> LiveSession | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        await socket.send_json({"type": "error", "message": "Invalid JSON."})
        return session
    if not isinstance(payload, dict):
        await socket.send_json({"type": "error", "message": "JSON messages must be objects."})
        return session

    kind = payload.get("type")
    if kind == "start":
        try:
            sample_rate = int(payload.get("sample_rate") or TARGET_SR)
        except (TypeError, ValueError):
            await socket.send_json({"type": "error", "message": "sample_rate must be an integer."})
            return session
        if not 8000 <= sample_rate <= 96000:
            await socket.send_json({"type": "error", "message": "sample_rate is out of range."})
            return session
        session = LiveSession(
            engine.transcribe,
            prompt_for=glossary.prompt_for,
            hotwords=glossary.current_hotwords,
            input_sample_rate=sample_rate,
            window_s=settings.window_s,
            partial_s=settings.partial_s,
            silence_rms=settings.silence_rms,
            id_prefix=f"s{next(serial)}",
        )
        glossary.refresh()
        await socket.send_json(
            {
                "type": "ready",
                "sample_rate": TARGET_SR,
                "encoding": "pcm_s16le",
                "channels": 1,
                "glossary_terms": len(glossary.terms),
                "window_s": settings.window_s,
            }
        )
        return session

    if kind == "stop":
        if session is None:
            await socket.send_json({"type": "error", "message": "Send start before stop."})
            return None
        try:
            events = await asyncio.to_thread(session.finish)
        except Exception:
            logger.exception("Live transcription failed")
            await socket.send_json({"type": "error", "message": "Transcription failed."})
            return None
        for event in events:
            await socket.send_json(event.to_json())
        await socket.send_json({"type": "done"})
        return None

    await socket.send_json({"type": "error", "message": f"Unknown message type {kind!r}."})
    return session


class _UploadCursor:
    """Decodes one window at a time. The engine lock is not held between windows."""

    def __init__(self, path: str, settings: Settings, model, glossary, prefix: str, duration: float | None) -> None:
        self.path = path
        self.settings = settings
        self.model = model
        self.glossary = glossary
        self.prefix = prefix
        self.duration = duration
        self.upload_id = prefix
        self.iterator = iter_pcm_windows(
            path,
            window_s=settings.batch_window_s,
            max_seconds=settings.max_audio_seconds,
        )
        self.done = False
        self.processed = 0.0
        self.produced = 0
        self.lock = threading.Lock()

    def pull(self) -> dict:
        with self.lock:
            if self.done:
                return self._payload(done=True, segments=[])
            try:
                samples, offset, duration = next(self.iterator)
            except StopIteration:
                self.done = True
                return self._payload(done=True, segments=[])
            rows = _window_segments(
                samples, offset, self.model, self.glossary, self.settings, self.prefix, self.produced
            )
            self.produced += len(rows)
            self.processed = offset + duration
            return self._payload(done=False, segments=[row.model_dump() for row in rows])

    def _payload(self, done: bool, segments: list) -> dict:
        return {
            "done": done,
            "segments": segments,
            "processed_seconds": round(self.processed, 3),
            "total_seconds": None if self.duration is None else round(self.duration, 3),
        }


def _discard_upload(uploads: dict, upload_id: str) -> None:
    cursor = uploads.pop(upload_id, None)
    if cursor is not None:
        _unlink(cursor.path)


def _too_long(exc: AudioTooLong) -> HTTPException:
    return HTTPException(
        status_code=413,
        detail={
            "code": "audio_too_long",
            "message": "录音太长，请分段上传",
            "max_seconds": int(exc.max_seconds),
        },
    )


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


async def _save_upload(file: UploadFile, limit: int, filename: str) -> str:
    async def chunks():
        while True:
            block = await file.read(1024 * 1024)
            if not block:
                break
            yield block

    return await _save_chunks(chunks(), limit, filename, None)


async def _save_chunks(chunks, limit: int, filename: str, declared: str | None) -> str:
    """Stream bytes to disk. Reject a negative or non-integer Content-Length."""

    if declared is not None and str(declared).strip() != "":
        try:
            announced = int(str(declared).strip())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Content-Length is invalid.") from exc
        if announced < 0:
            raise HTTPException(status_code=400, detail="Content-Length is invalid.")
        if announced > limit:
            raise HTTPException(status_code=413, detail="Recording is larger than the upload limit.")
    suffix = os.path.splitext(filename)[1].lower() or ".bin"
    handle = tempfile.NamedTemporaryFile(prefix="asr-upload-", suffix=suffix, delete=False)
    path = handle.name
    total = 0
    try:
        async for chunk in chunks:
            total += len(chunk)
            if total > limit:
                raise HTTPException(status_code=413, detail="Recording is larger than the upload limit.")
            handle.write(chunk)
    except Exception:
        handle.close()
        _unlink(path)
        raise
    handle.close()
    if total == 0:
        _unlink(path)
        raise HTTPException(status_code=400, detail="The upload was empty.")
    return path


def _transcribe_file(path: str, settings: Settings, model, glossary, prefix: str) -> list[SegmentOut]:
    segments: list[SegmentOut] = []
    try:
        for samples, offset, _duration in iter_pcm_windows(
            path,
            window_s=settings.batch_window_s,
            max_seconds=settings.max_audio_seconds,
        ):
            segments.extend(_window_segments(samples, offset, model, glossary, settings, prefix, len(segments)))
    except AudioTooLong:
        segments.clear()
        raise
    return segments


def _window_segments(samples, offset: float, model, glossary, settings: Settings, prefix: str, already: int) -> list[SegmentOut]:
    # The engine lock is taken and released inside transcribe, one window at a time.
    parts = model.transcribe(
        samples,
        TARGET_SR,
        glossary.prompt_for(""),
        glossary.current_hotwords(),
        vad=settings.batch_vad,
    )
    rows: list[SegmentOut] = []
    for index, part in enumerate(parts):
        rows.append(
            SegmentOut(
                id=f"{prefix}-seg-{already + index + 1}",
                start=round(offset + part.start, 3),
                end=round(offset + part.end, 3),
                text=part.text,
            )
        )
    return rows


def _configure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
