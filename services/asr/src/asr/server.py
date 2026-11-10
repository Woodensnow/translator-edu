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
import json
import logging
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from asr.audio import TARGET_SR, AudioFormatError, decode_audio_bytes
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
        data = await file.read(resolved.max_upload_bytes + 1)
        if len(data) > resolved.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Recording is larger than the upload limit.")
        if not data:
            raise HTTPException(status_code=400, detail="The upload was empty.")

        def run() -> list:
            try:
                samples, sample_rate = decode_audio_bytes(data, file.filename or "")
            except AudioFormatError:
                raise
            if samples.size == 0:
                raise AudioFormatError("The audio file had no samples.")
            return model.transcribe(
                samples,
                sample_rate,
                glossary.prompt_for(""),
                glossary.current_hotwords(),
                vad=resolved.batch_vad,
            )

        try:
            segments = await asyncio.to_thread(run)
        except AudioFormatError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Batch transcription failed")
            raise HTTPException(status_code=500, detail="Transcription failed.") from exc

        payload = [
            SegmentOut(
                id=f"seg-{index + 1}",
                start=round(segment.start, 3),
                end=round(segment.end, 3),
                text=segment.text,
            )
            for index, segment in enumerate(segments)
        ]
        return TranscriptOut(language="en", text=" ".join(item.text for item in payload), segments=payload)

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


def _configure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
