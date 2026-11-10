"""HTTP API for English-to-Chinese lecture captions.

This README-level contract lives in ``services/translate/README.md``. Unknown
JSON fields are ignored so a later shared contract can add fields.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app import __version__
from app.config import Settings
from app.engine import translate_texts
from app.glossary import GlossaryFormatError, GlossaryNotFound, GlossaryStore
from app.translator import ModelNotReady, build_translator

logger = logging.getLogger("translate.api")

_ENGLISH = {"en", "eng", "english"}
_CHINESE = {"zh", "zho", "cmn", "chi", "chinese", "zh-cn", "zh-hans"}


class RequestError(ValueError):
    """The caller sent a payload this service will not translate."""


class SegmentIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)
    text: str

    @model_validator(mode="after")
    def _check_span(self) -> SegmentIn:
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("end is before start")
        return self


class TranslateRequest(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "examples": [
                {
                    "glossary": "deep-learning",
                    "segments": [
                        {
                            "id": "seg-1",
                            "start": 12.4,
                            "end": 15.1,
                            "text": "Backpropagation uses gradient descent.",
                        }
                    ],
                }
            ]
        },
    )

    text: str | None = None
    segments: list[SegmentIn] | None = None
    glossary: str | None = None
    source_lang: str = "en"
    target_lang: str = "zh"

    @field_validator("source_lang", "target_lang", mode="before")
    @classmethod
    def _normalize_lang(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower().replace("_", "-")
        return value

    @field_validator("glossary", mode="before")
    @classmethod
    def _normalize_glossary(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @model_validator(mode="after")
    def _check_payload(self) -> TranslateRequest:
        if self.text is None and not self.segments:
            raise ValueError("provide text or segments")
        if self.source_lang not in _ENGLISH and not self.source_lang.startswith("en-"):
            raise ValueError("source_lang must be English")
        if self.target_lang not in _CHINESE and not self.target_lang.startswith("zh-"):
            raise ValueError("target_lang must be Chinese")
        return self


class TermOut(BaseModel):
    en: str
    zh: str
    note: str
    matched: str


class SegmentOut(BaseModel):
    id: str
    start: float | None
    end: float | None
    source: str
    translation: str
    terms: list[TermOut]


class TranslateResponse(BaseModel):
    source_lang: str
    target_lang: str
    glossary: str
    model: str
    backend: str
    translation: str
    segments: list[SegmentOut]


@dataclass(frozen=True)
class _Incoming:
    id: str | None
    start: float | None
    end: float | None
    text: str


def _incoming(body: TranslateRequest) -> list[_Incoming]:
    if body.segments:
        return [_Incoming(segment.id, segment.start, segment.end, segment.text) for segment in body.segments]
    return [_Incoming(None, None, None, body.text or "")]


def create_app(
    settings: Settings | None = None,
    translator: object | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    backend = translator or build_translator(settings)
    store = GlossaryStore(settings.glossary_dir)

    def _translate_sync(body: TranslateRequest) -> TranslateResponse:
        units = _incoming(body)
        if len(units) > settings.max_segments:
            raise RequestError(f"at most {settings.max_segments} segments are accepted")
        if sum(len(unit.text) for unit in units) > settings.max_chars:
            raise RequestError(f"at most {settings.max_chars} characters are accepted")
        glossary_name = body.glossary or settings.default_glossary
        glossary = store.get(glossary_name)
        results = translate_texts(
            [unit.text for unit in units],
            glossary,
            backend,
            max_unit_chars=settings.max_unit_chars,
        )
        segments: list[SegmentOut] = []
        for index, (unit, result) in enumerate(zip(units, results)):
            segments.append(
                SegmentOut(
                    id=unit.id or f"s{index}",
                    start=unit.start,
                    end=unit.end,
                    source=result.source,
                    translation=result.translation,
                    terms=[
                        TermOut(en=term.en, zh=term.zh, note=term.note, matched=term.matched)
                        for term in result.terms
                    ],
                )
            )
        return TranslateResponse(
            source_lang="en",
            target_lang="zh",
            glossary=glossary.name,
            model=settings.model_name,
            backend=backend.name,
            translation="\n".join(segment.translation for segment in segments),
            segments=segments,
        )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            names = store.names()
        except GlossaryFormatError as exc:
            logger.warning("glossary not loaded: %s", exc)
        else:
            logger.info("glossaries: %s", ", ".join(names))
        ready, detail = backend.health()
        if ready:
            logger.info("translation backend %s is ready", backend.name)
        else:
            logger.warning("translation backend %s is not ready: %s", backend.name, detail)
        yield

    app = FastAPI(
        title="Lecture translation",
        version=__version__,
        summary="English to Chinese captions with a course glossary.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        glossary_error: str | None = None
        names: list[str] = []
        try:
            names = store.names()
        except GlossaryFormatError as exc:
            glossary_error = str(exc)
        translator_ready, translator_detail = backend.health()
        ready = translator_ready and glossary_error is None
        details = [part for part in (glossary_error, translator_detail) if part]
        payload = {
            "status": "ok" if ready else "not_ready",
            "ready": ready,
            "backend": backend.name,
            "model": settings.model_name,
            "version": __version__,
            "glossaries": names,
            "default_glossary": settings.default_glossary,
            "detail": "; ".join(details) if details else None,
        }
        return JSONResponse(status_code=200 if ready else 503, content=payload)

    @app.get("/glossary")
    def list_glossaries() -> dict[str, object]:
        try:
            items = store.describe()
        except GlossaryFormatError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "default": settings.default_glossary,
            "columns": ["en", "zh", "note"],
            "glossaries": items,
        }

    @app.get("/glossary/{name}")
    def get_glossary(name: str) -> dict[str, object]:
        try:
            glossary = store.get(name)
        except GlossaryNotFound as exc:
            raise HTTPException(
                status_code=404,
                detail={"message": str(exc), "glossaries": exc.available},
            ) from exc
        except GlossaryFormatError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "name": glossary.name,
            "file": glossary.source.name,
            "columns": ["en", "zh", "note"],
            "terms": [{"en": term.en, "zh": term.zh, "note": term.note} for term in glossary.terms],
        }

    @app.post("/translate", response_model=TranslateResponse)
    async def translate(body: TranslateRequest) -> TranslateResponse:
        try:
            return await asyncio.to_thread(_translate_sync, body)
        except GlossaryNotFound as exc:
            raise HTTPException(
                status_code=404,
                detail={"message": str(exc), "glossaries": exc.available},
            ) from exc
        except RequestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelNotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except GlossaryFormatError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


app = create_app()
