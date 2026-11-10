"""Course glossary loaded from shared CSV files.

Matching prefers the longest phrase and ignores case. A hyphen in a term also
matches a space or nothing (``fine-tuning`` / ``fine tuning`` / ``finetuning``),
and a trailing ``s`` or ``es`` covers plurals (``mini-batch`` / ``mini-batches``).

Hits become sentinels the translator must pass through; the Chinese term is
written back afterwards. The sentinel shape was chosen by
``scripts/benchmark_sentinels.py`` so it usually survives opus-mt-en-zh.
See ``engine`` for the fallback that does not depend on that.
"""

from __future__ import annotations

import csv
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("translate.glossary")

# Uppercase code plus digits. Measured on the int8 CPU model; see
# benchmarks/sentinel-results.md. Bracket sentinels such as ⟦G0⟧ are not in
# the SentencePiece vocab and were dropped.
SENTINEL_PREFIX = "ZXQ"
PLACEHOLDER_PATTERN = re.compile(rf"(?<![A-Za-z]){re.escape(SENTINEL_PREFIX)}(\d+)", re.IGNORECASE)
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
    return f"{SENTINEL_PREFIX}{index}"


def _alloc_index(used: set[int]) -> int:
    index = 0
    while index in used:
        index += 1
    used.add(index)
    return index


_TITLE_NAME = re.compile(r"^[A-Z][a-z]+$")


def term_pattern(term: str) -> re.Pattern[str]:
    """Whole-phrase match.

    Hyphens match a hyphen, a space, or nothing. Spaces match whitespace.
    Ordinary terms accept a trailing ``s`` or ``es`` (``mini-batches``, ``biases``).
    All-caps acronyms match only in that case, plus a lowercase plural ``s``
    (``AIs``, ``CNNs``), so ``ais`` is not locked as AI. A title-case name such
    as ``Adam`` does not take a plural, so ``Adams`` is left alone.
    """
    body = re.escape(term).replace(r"\ ", r"\s+").replace(r"\-", r"[\s-]?")
    prefix = r"(?<![A-Za-z0-9])"
    boundary = r"(?![A-Za-z0-9])"
    if term.isupper() and any(character.isalpha() for character in term):
        return re.compile(rf"{prefix}{body}s?{boundary}")
    if _TITLE_NAME.fullmatch(term):
        return re.compile(rf"{prefix}{body}{boundary}", re.IGNORECASE)
    return re.compile(rf"{prefix}{body}(?:es|s)?{boundary}", re.IGNORECASE)


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
            if any(char in en or char in zh for char in "\n\r"):
                raise GlossaryFormatError(f"{path}:{reader.line_num}: terms cannot contain newlines")
            if SENTINEL_PREFIX.casefold() in en.casefold() or SENTINEL_PREFIX.casefold() in zh.casefold():
                raise GlossaryFormatError(
                    f"{path}:{reader.line_num}: terms cannot contain the glossary sentinel"
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
        used = {int(number) for number in PLACEHOLDER_PATTERN.findall(text)}
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
                placeholder=placeholder_for(_alloc_index(used)),
                en=term.en,
                zh=term.zh,
                note=term.note,
                matched=text[start:end],
            )
            for start, end, term in chosen
        ]
        rendered = text
        for match in reversed(matches):
            rendered = rendered[: match.start] + match.placeholder + rendered[match.end :]
        return rendered, matches


def _placeholder_regex(placeholder: str) -> re.Pattern[str]:
    """Match a sentinel, including the Z→兹 transliteration this model emits."""
    if not placeholder.startswith(SENTINEL_PREFIX):
        raise ValueError(f"unexpected sentinel {placeholder}")
    number = placeholder[len(SENTINEL_PREFIX) :]
    if not number.isdigit():
        raise ValueError(f"unexpected sentinel {placeholder}")
    return re.compile(
        rf"(?<![A-Za-z])(?:{re.escape(SENTINEL_PREFIX)}|兹XQ)\s*{number}(?!\d)",
        re.IGNORECASE,
    )


def restore_placeholders(text: str, matches: list[Match]) -> tuple[str, list[Match]]:
    """Swap sentinels for Chinese. Returns matches whose sentinel disappeared.

    Matching is case-insensitive and allows spaces before the digits
    (``ZXQ 0``). On this checkpoint a leading ``Z`` is sometimes translated
    as ``兹`` (``兹XQ0``), and that form counts too.
    """
    missing: list[Match] = []
    rendered = text
    for match in sorted(matches, key=lambda item: len(item.placeholder), reverse=True):
        pattern = _placeholder_regex(match.placeholder)
        updated, count = pattern.subn(lambda _match, zh=match.zh: zh, rendered)
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
