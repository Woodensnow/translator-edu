"""Course glossary loader for ASR prompting.

The translate service owns glossary files. They live at ``glossary/<course>.csv``.
The first course file is ``glossary/deep-learning.csv``, which Anna commits.
This package does not create that file.

``ASR_GLOSSARY_PATH`` overrides the path. The default is
``glossary/deep-learning.csv``. A missing file is not fatal: the loader logs a
warning and returns no terms so transcription still starts.

Accepted headers (English names are case-insensitive):

- ``en,zh,note``
- ``英文,中文,备注``

The notes column may be empty. A file with only ``en,zh`` (or ``英文,中文``) is
accepted and treated as an empty note. Only the English column is used for the
Whisper initial prompt and hotwords. Chinese text and the notes column are
never copied into the prompt.

This module is the only CSV parser. If the shared format changes, update this
file and ``docs/glossary.md`` together.
"""

from __future__ import annotations

import csv
import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)

DEFAULT_GLOSSARY_PATH = "glossary/deep-learning.csv"

_HEADER_MAP = {
    "en": "en",
    "english": "en",
    "英文": "en",
    "zh": "zh",
    "chinese": "zh",
    "中文": "zh",
    "note": "note",
    "notes": "note",
    "备注": "note",
}


class GlossaryFormatError(ValueError):
    """The CSV exists but does not use a known glossary header."""


@dataclass(frozen=True)
class GlossaryEntry:
    en: str
    zh: str
    note: str


def glossary_path_from_env(env: Mapping[str, str] | None = None) -> Path:
    """Resolve ``ASR_GLOSSARY_PATH`` without reading the file."""

    source = os.environ if env is None else env
    raw = source.get("ASR_GLOSSARY_PATH", DEFAULT_GLOSSARY_PATH)
    return Path(str(raw).strip() or DEFAULT_GLOSSARY_PATH)


def load_glossary(path: str | Path) -> list[GlossaryEntry]:
    """Load glossary rows. Missing or empty files yield an empty list."""

    csv_path = Path(path)
    if not csv_path.is_file():
        logger.warning(
            "Glossary file not found at %s; continuing with no terms. "
            "Set ASR_GLOSSARY_PATH (default %s). "
            "Expected a CSV header of 'en,zh,note' or '英文,中文,备注'. "
            "Per-course files live under glossary/ and are committed by the translate owner.",
            csv_path,
            DEFAULT_GLOSSARY_PATH,
        )
        return []

    text = csv_path.read_text(encoding="utf-8-sig")
    if not text.strip():
        logger.warning("Glossary file %s is empty; continuing with no terms.", csv_path)
        return []

    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        return []

    columns = [_map_header(cell) for cell in header]
    if columns.count("en") != 1 or columns.count("zh") != 1 or columns.count("note") > 1:
        raise GlossaryFormatError(
            f"{csv_path} header {header!r} is not a glossary. "
            "Expected en,zh,note or 英文,中文,备注."
        )

    en_i = columns.index("en")
    zh_i = columns.index("zh")
    note_i = columns.index("note") if "note" in columns else None
    entries: list[GlossaryEntry] = []
    for row in reader:
        if not row or all(not cell.strip() for cell in row):
            continue
        en = row[en_i].strip() if en_i < len(row) else ""
        if not en:
            continue
        zh = row[zh_i].strip() if zh_i < len(row) else ""
        note = row[note_i].strip() if note_i is not None and note_i < len(row) else ""
        entries.append(GlossaryEntry(en=en, zh=zh, note=note))
    return entries


def load_english_terms(path: str | Path) -> list[str]:
    """English column only, in file order, with duplicates removed."""

    return english_terms(load_glossary(path))


class GlossaryPrompt:
    """English prompt terms. Reloads when the CSV's mtime changes.

    The translate service hot-reloads ``glossary/<course>.csv``. ASR reads the
    same mounted file; checking mtime keeps a long-lived process on the edited
    list without restarting, and without parsing the CSV on every audio frame.
    """

    def __init__(self, path: str | Path, max_chars: int = 500, hotword_max_chars: int = 400) -> None:
        self.path = Path(path)
        self.max_chars = max_chars
        self.hotword_max_chars = hotword_max_chars
        self._mtime: float | None = None
        self._loaded = False
        self.terms: list[str] = []
        self.hotwords: str | None = None

    def refresh(self) -> None:
        mtime = self.path.stat().st_mtime if self.path.is_file() else None
        if self._loaded and mtime == self._mtime:
            return
        self.terms = load_english_terms(self.path)
        self.hotwords = build_hotwords(self.terms, max_chars=self.hotword_max_chars)
        self._mtime = mtime
        self._loaded = True
        if self.terms and self.hotwords:
            kept = self.hotwords.count(",") + 1
            if kept < len(self.terms):
                logger.warning(
                    "Glossary hotword list truncated from %d to %d terms to fit the prompt budget",
                    len(self.terms),
                    kept,
                )

    def prompt_for(self, previous: str) -> str:
        self.refresh()
        return build_initial_prompt(self.terms, previous=previous, max_chars=self.max_chars)

    def current_hotwords(self) -> str | None:
        self.refresh()
        return self.hotwords


def english_terms(entries: list[GlossaryEntry]) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for entry in entries:
        if entry.en not in seen:
            seen.add(entry.en)
            terms.append(entry.en)
    return terms


def build_initial_prompt(
    terms: list[str],
    previous: str = "",
    max_chars: int = 500,
) -> str:
    """Build a Whisper initial prompt from English terms and recent context.

    Chinese glossary text is never included. Terms that do not fit are dropped
    from the end of the list. ``previous`` is the last finalized English line,
    clipped to the remaining character budget.
    """

    if max_chars <= 0:
        return ""
    base = "English technical lecture."
    if len(base) >= max_chars:
        return base[:max_chars]

    previous = " ".join((previous or "").split())
    chosen: list[str] = []
    for term in terms:
        cleaned = " ".join(term.split())
        if not cleaned:
            continue
        candidate = chosen + [cleaned]
        sentence = _term_sentence(candidate)
        tail = f" {previous}" if previous else ""
        if len(base) + len(sentence) + len(tail) <= max_chars:
            chosen.append(cleaned)
            continue
        # Keep terms over previous context when the budget is tight.
        if len(base) + len(sentence) <= max_chars:
            chosen.append(cleaned)
            previous = ""
            continue
        break

    sentence = _term_sentence(chosen)
    room = max_chars - len(base) - len(sentence)
    tail = ""
    if previous and room > 1:
        clipped = previous[: room - 1].strip()
        if clipped:
            tail = " " + clipped
    return (base + sentence + tail)[:max_chars]


def build_hotwords(terms: list[str], max_terms: int = 80, max_chars: int = 400) -> str | None:
    """Comma-separated English terms for faster-whisper's ``hotwords`` argument."""

    if max_terms <= 0 or max_chars <= 0:
        return None
    chosen: list[str] = []
    used = 0
    for term in terms:
        cleaned = " ".join(term.split())
        if not cleaned:
            continue
        extra = len(cleaned) if not chosen else len(cleaned) + 2
        if len(chosen) >= max_terms or used + extra > max_chars:
            break
        chosen.append(cleaned)
        used += extra
    if not chosen:
        return None
    return ", ".join(chosen)


def _term_sentence(terms: list[str]) -> str:
    if not terms:
        return ""
    return " Terms: " + ", ".join(terms) + "."


def _map_header(cell: str) -> str:
    key = cell.strip().lstrip("\ufeff")
    return _HEADER_MAP.get(key) or _HEADER_MAP.get(key.lower()) or ""
