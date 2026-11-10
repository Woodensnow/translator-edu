"""Course glossary loaded from shared CSV files.

Matching prefers the longest phrase, ignores case, and accepts a simple English
plural. Hits become sentinels the translator must pass through; the Chinese
term is written back afterwards. See ``engine`` for the fallback that does not
depend on the sentinel surviving the model.
"""

from __future__ import annotations

import csv
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("translate.glossary")

PLACEHOLDER_PATTERN = re.compile(r"⟦G\d+⟧")
REQUIRED_COLUMNS = ("en", "zh", "note")


class GlossaryFormatError(ValueError):
    """A glossary CSV is missing, unreadable, or not in the shared shape."""


class GlossaryNotFound(LookupError):
    def __init__(self, name: str, available: list[str]) -> None:
        super().__init__(f"unknown glossary {name!r}")
        self.name = name
        self.available = available


@dataclass(frozen=True)
class Term:
    en: str
    zh: str
    note: str


@dataclass(frozen=True)
class Match:
    start: int
    end: int
    placeholder: str
    en: str
    zh: str
    note: str
    matched: str


def placeholder_for(index: int) -> str:
    if index < 0:
        raise ValueError("placeholder index must be non-negative")
    return f"⟦G{index}⟧"


def term_pattern(term: str) -> re.Pattern[str]:
    """Whole-phrase match. A final ``s`` or ``es`` covers ordinary plurals."""
    body = re.escape(term).replace(r"\ ", r"\s+")
    suffix = r"(?:es)?" if term[-1].casefold() == "s" else r"s?"
    return re.compile(
        rf"(?<![A-Za-z0-9]){body}{suffix}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


def load_glossary_file(path: Path) -> Glossary:
    try:
        handle = path.open(encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise GlossaryFormatError(f"cannot read {path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        columns = [name.strip() for name in fieldnames]
        missing = [name for name in REQUIRED_COLUMNS if name not in columns]
        if missing:
            raise GlossaryFormatError(
                f"{path}: header must include {', '.join(REQUIRED_COLUMNS)}"
            )
        terms: list[Term] = []
        seen: set[str] = set()
        for row in reader:
            en = " ".join((row.get("en") or "").split())
            zh = " ".join((row.get("zh") or "").split())
            note = (row.get("note") or "").strip()
            if not en and not zh and not note:
                continue
            if not en or not zh:
                raise GlossaryFormatError(f"{path}:{reader.line_num}: en and zh are required")
            if any(char in en or char in zh for char in "⟦⟧\n\r"):
                raise GlossaryFormatError(
                    f"{path}:{reader.line_num}: terms cannot contain glossary sentinels"
                )
            key = en.casefold()
            if key in seen:
                raise GlossaryFormatError(f"{path}:{reader.line_num}: duplicate term {en}")
            seen.add(key)
            terms.append(Term(en, zh, note))
    if not terms:
        raise GlossaryFormatError(f"{path}: no terms")
    return Glossary(path.stem, path, terms)


def load_glossary_directory(directory: Path) -> dict[str, Glossary]:
    if not directory.is_dir():
        raise GlossaryFormatError(f"glossary directory not found: {directory}")
    files = sorted(path for path in directory.glob("*.csv") if path.is_file())
    if not files:
        raise GlossaryFormatError(f"no CSV glossaries in {directory}")
    glossaries: dict[str, Glossary] = {}
    folds: dict[str, str] = {}
    for path in files:
        glossary = load_glossary_file(path)
        fold = glossary.name.casefold()
        if fold in folds:
            raise GlossaryFormatError(
                f"glossary names {folds[fold]!r} and {glossary.name!r} collide"
            )
        folds[fold] = glossary.name
        glossaries[glossary.name] = glossary
    return glossaries


class Glossary:
    def __init__(self, name: str, source: Path, terms: list[Term]) -> None:
        self.name = name
        self.source = source
        self.terms = tuple(terms)
        self._compiled = tuple((term, term_pattern(term.en)) for term in self.terms)

    def protect(self, text: str) -> tuple[str, list[Match]]:
        """Replace glossary hits with sentinels. Spans refer to ``text``."""
        if not text:
            return text, []
        candidates: list[tuple[int, int, Term]] = []
        for term, pattern in self._compiled:
            for found in pattern.finditer(text):
                candidates.append((found.start(), found.end(), term))
        candidates.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[2].en.casefold()))
        chosen: list[tuple[int, int, Term]] = []
        occupied: list[tuple[int, int]] = []
        for start, end, term in candidates:
            if any(start < prev_end and end > prev_start for prev_start, prev_end in occupied):
                continue
            chosen.append((start, end, term))
            occupied.append((start, end))
        chosen.sort(key=lambda item: item[0])
        matches = [
            Match(
                start=start,
                end=end,
                placeholder=placeholder_for(index),
                en=term.en,
                zh=term.zh,
                note=term.note,
                matched=text[start:end],
            )
            for index, (start, end, term) in enumerate(chosen)
        ]
        rendered = text
        for match in reversed(matches):
            rendered = rendered[: match.start] + match.placeholder + rendered[match.end :]
        return rendered, matches


def restore_placeholders(text: str, matches: list[Match]) -> tuple[str, list[Match]]:
    """Swap sentinels for Chinese. Returns matches whose sentinel disappeared."""
    missing: list[Match] = []
    rendered = text
    for match in matches:
        if match.placeholder in rendered:
            rendered = rendered.replace(match.placeholder, match.zh)
            continue
        number = match.placeholder.removeprefix("⟦G").removesuffix("⟧")
        loose = re.compile(rf"⟦\s*G\s*{re.escape(number)}\s*⟧")
        updated, count = loose.subn(match.zh, rendered)
        if count:
            rendered = updated
        else:
            missing.append(match)
    return rendered, missing


class GlossaryStore:
    """Load every ``*.csv`` in a directory and reload when a file changes.

    A broken save does not drop a glossary that previously loaded. The first
    load must succeed.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._lock = threading.Lock()
        self._glossaries: dict[str, Glossary] = {}
        self._good_signature: tuple[tuple[str, int, int], ...] | None = None
        self._bad_signature: tuple[tuple[str, int, int], ...] | None = None
        self._last_error: str | None = None

    def names(self) -> list[str]:
        with self._lock:
            self._ensure_loaded()
            return sorted(self._glossaries)

    def get(self, name: str) -> Glossary:
        with self._lock:
            self._ensure_loaded()
            for glossary in self._glossaries.values():
                if glossary.name.casefold() == name.casefold():
                    return glossary
            raise GlossaryNotFound(name, sorted(self._glossaries))

    def describe(self) -> list[dict[str, object]]:
        with self._lock:
            self._ensure_loaded()
            items = []
            for glossary in sorted(self._glossaries.values(), key=lambda item: item.name):
                items.append(
                    {
                        "name": glossary.name,
                        "terms": len(glossary.terms),
                        "file": glossary.source.name,
                    }
                )
            return items

    def _signature(self) -> tuple[tuple[str, int, int], ...]:
        if not self.directory.is_dir():
            raise GlossaryFormatError(f"glossary directory not found: {self.directory}")
        files = sorted(path for path in self.directory.glob("*.csv") if path.is_file())
        if not files:
            raise GlossaryFormatError(f"no CSV glossaries in {self.directory}")
        return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in files)

    def _ensure_loaded(self) -> None:
        try:
            signature = self._signature()
        except (GlossaryFormatError, OSError) as exc:
            if self._glossaries:
                message = str(exc)
                if message != self._last_error:
                    logger.warning("glossary reload failed; keeping the previous terms: %s", message)
                    self._last_error = message
                return
            raise
        if signature == self._good_signature:
            return
        if signature == self._bad_signature and self._glossaries:
            return
        try:
            loaded = load_glossary_directory(self.directory)
        except GlossaryFormatError as exc:
            if self._glossaries:
                message = str(exc)
                if message != self._last_error:
                    logger.warning(
                        "glossary reload failed; keeping the previous terms: %s", message
                    )
                    self._last_error = message
                self._bad_signature = signature
                return
            raise
        self._glossaries = loaded
        self._good_signature = signature
        self._bad_signature = None
        self._last_error = None
