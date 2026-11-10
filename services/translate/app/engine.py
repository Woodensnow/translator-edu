"""Translate text with glossary terms locked around the model.

Each hit is replaced with a sentinel, the protected string is translated, and
the sentinel is swapped for the Chinese term. If a sentinel does not survive
the model, the original sentence is translated in pieces around those terms so
the Chinese is still inserted. Terminology does not depend on the model alone.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol

from app.glossary import PLACEHOLDER_PATTERN, Glossary, Match, restore_placeholders

logger = logging.getLogger("translate.engine")

_SENTENCE_BOUNDARY = re.compile(r"[.!?。！？]\s+|\n+")


class TextTranslator(Protocol):
    def translate_many(self, texts: list[str]) -> list[str]: ...


@dataclass(frozen=True)
class AppliedTerm:
    en: str
    zh: str
    note: str
    matched: str


@dataclass(frozen=True)
class UnitResult:
    source: str
    translation: str
    terms: tuple[AppliedTerm, ...]


def split_for_translation(text: str, limit: int) -> list[str]:
    """Split long text without tearing a glossary sentinel.

    The pieces concatenate back to ``text``. Short captions are left whole.
    """
    if limit < 32:
        raise ValueError("max_unit_chars must be at least 32")
    if len(text) <= limit:
        return [text]
    spans = [(match.start(), match.end()) for match in PLACEHOLDER_PATTERN.finditer(text)]

    def blocked(cut: int) -> bool:
        return any(start < cut < end for start, end in spans)

    parts: list[str] = []
    start = 0
    while len(text) - start > limit:
        window_end = min(len(text), start + limit)
        cut: int | None = None
        window = text[start:window_end]
        for boundary in _SENTENCE_BOUNDARY.finditer(window):
            candidate = start + boundary.end()
            if candidate > start and not blocked(candidate):
                cut = candidate
        if cut is None:
            for index in range(window_end, start, -1):
                if text[index - 1].isspace() and not blocked(index):
                    cut = index
                    break
        if cut is None or cut <= start:
            cut = window_end
            while cut < len(text) and blocked(cut):
                cut += 1
            if cut <= start:
                cut = min(len(text), start + limit)
        parts.append(text[start:cut])
        start = cut
    parts.append(text[start:])
    return parts


def _needs_translation(chunk: str) -> bool:
    return any(character.isalpha() for character in chunk)


def splice_translate(
    text: str,
    matches: list[Match],
    translate_many: TextTranslator,
) -> str:
    """Translate the words around locked terms and keep the glossary Chinese."""
    ordered = sorted(matches, key=lambda item: item.start)
    pieces: list[tuple[str, str]] = []
    cursor = 0
    for match in ordered:
        if match.start < cursor:
            raise RuntimeError("overlapping glossary matches")
        if match.start > cursor:
            pieces.append(("text", text[cursor : match.start]))
        pieces.append(("term", match.zh))
        cursor = match.end
    if cursor < len(text):
        pieces.append(("text", text[cursor:]))

    pending_at: list[int] = []
    pending_text: list[str] = []
    for index, (kind, value) in enumerate(pieces):
        if kind == "text" and _needs_translation(value):
            pending_at.append(index)
            pending_text.append(value)
    translated = list(translate_many.translate_many(pending_text)) if pending_text else []
    if len(translated) != len(pending_text):
        raise RuntimeError("translator returned a different number of segments")

    rendered: list[str] = []
    cursor_translated = 0
    pending = set(pending_at)
    for index, (kind, value) in enumerate(pieces):
        if kind == "term":
            rendered.append(value)
        elif index in pending:
            rendered.append(translated[cursor_translated])
            cursor_translated += 1
        else:
            rendered.append(value)
    return "".join(rendered)


def translate_texts(
    texts: list[str],
    glossary: Glossary,
    translator: TextTranslator,
    max_unit_chars: int = 480,
) -> list[UnitResult]:
    plans: list[tuple[str, list[Match], int, int]] = []
    flat: list[str] = []
    for text in texts:
        protected, matches = glossary.protect(text)
        parts = split_for_translation(protected, max_unit_chars)
        start = len(flat)
        flat.extend(parts)
        plans.append((text, matches, start, start + len(parts)))

    translated = list(translator.translate_many(flat)) if flat else []
    if len(translated) != len(flat):
        raise RuntimeError("translator returned a different number of segments")

    results: list[UnitResult] = []
    for text, matches, start, end in plans:
        joined = "".join(translated[start:end])
        if matches:
            restored, missing = restore_placeholders(joined, matches)
            if missing:
                logger.info(
                    "glossary sentinel missing for %s term(s); translating around locked terms",
                    len(missing),
                )
                restored = splice_translate(text, matches, translator)
        else:
            restored = joined
        results.append(
            UnitResult(
                source=text,
                translation=restored,
                terms=tuple(
                    AppliedTerm(match.en, match.zh, match.note, match.matched) for match in matches
                ),
            )
        )
    return results
