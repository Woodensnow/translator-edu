"""Finished upload results on disk.

Audio is not stored. One JSON file per completed job holds the bilingual
segments. The write is a temp file in the same directory plus ``os.replace``,
so a reader never sees a partial document. Job ids are twelve lowercase hex
characters; anything else is rejected before it is used as a path.

``RESULT_RETENTION_DAYS`` defaults to 7. ``0`` keeps files until a person
deletes them. Expired files are removed on startup and on a timer.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
import time
from pathlib import Path

JOB_ID_RE = re.compile(r"[0-9a-f]{12}")
DEFAULT_RESULT_DIR = "/data/results"
DEFAULT_RETENTION_DAYS = 7
DEFAULT_PRUNE_INTERVAL = 3600.0


class InvalidJobId(ValueError):
    """The id is not a safe result filename."""


def parse_retention_days(raw: str | None) -> int:
    if raw is None or str(raw).strip() == "":
        return DEFAULT_RETENTION_DAYS
    try:
        days = int(str(raw).strip())
    except ValueError as exc:
        raise ValueError("RESULT_RETENTION_DAYS must be an integer.") from exc
    if days < 0:
        raise ValueError("RESULT_RETENTION_DAYS must be >= 0.")
    return days


def public_segment(segment: object) -> dict | None:
    if not isinstance(segment, dict):
        return None
    terms = segment.get("terms")
    return {
        "id": segment.get("id"),
        "start": segment.get("start") or 0,
        "end": segment.get("end") or 0,
        "text": "" if segment.get("text") is None else str(segment.get("text")),
        "translation": "" if segment.get("translation") is None else str(segment.get("translation")),
        "fallback": bool(segment.get("fallback")),
        "terms": list(terms) if isinstance(terms, list) else [],
    }


def snapshot_payload(job) -> dict:
    """Copy a finished job. The caller holds ``job.lock`` when the job can change."""

    segments = []
    for segment in job.segments:
        copied = public_segment(segment)
        if copied is not None:
            segments.append(copied)
    return {
        "id": job.id,
        "name": job.filename or "lecture",
        "filename": job.filename or "lecture",
        "finished_at": float(job.finished_at or 0),
        "duration": job.total_seconds,
        "segments": segments,
    }


def _js_round(value: float) -> int:
    """Match ``Math.round`` for the caption timestamps the page already writes."""

    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def format_timestamp(seconds) -> str:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        value = 0.0
    if not math.isfinite(value) or value <= 0:
        value = 0.0
    total_tenths = _js_round(value * 10)
    tenths = total_tenths % 10
    whole = total_tenths // 10
    minutes = whole // 60
    secs = whole % 60
    return f"{minutes}:{secs:02d}.{tenths}"


def render_txt(segments) -> str:
    blocks = []
    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        blocks.append(
            f"{format_timestamp(segment.get('start'))}\n{segment.get('text') or ''}\n{segment.get('translation') or ''}"
        )
    return "\n\n".join(blocks)


def srt_time(seconds) -> str:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        value = 0.0
    if not math.isfinite(value) or value < 0:
        value = 0.0
    ms = _js_round(value * 1000)
    hours = ms // 3_600_000
    minutes = (ms % 3_600_000) // 60_000
    secs = (ms % 60_000) // 1000
    milli = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milli:03d}"


def render_srt(segments) -> str:
    blocks = []
    index = 1
    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        end_value = segment.get("end")
        if end_value is None:
            end_value = segment.get("start") or 0
        lines = [str(segment.get("text") or ""), str(segment.get("translation") or "")]
        lines = [line for line in lines if line.strip()]
        blocks.append(f"{index}\n{srt_time(segment.get('start') or 0)} --> {srt_time(end_value)}\n" + "\n".join(lines))
        index += 1
    return "\n\n".join(blocks)


def write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` via a sibling temp file and ``os.replace``."""

    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=directory)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        dir_fd = os.open(directory, flags)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


class ResultStore:
    def __init__(self, directory=None, retention_days=None, clock=time.time, prune_interval=None) -> None:
        self._directory = None if directory is None else Path(directory)
        self._retention_days = retention_days
        self._prune_interval = prune_interval
        self.clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()

    @property
    def directory(self) -> Path:
        if self._directory is not None:
            return self._directory
        return Path(os.environ.get("RESULT_DIR", DEFAULT_RESULT_DIR))

    @property
    def retention_days(self) -> int:
        if self._retention_days is not None:
            return int(self._retention_days)
        return parse_retention_days(os.environ.get("RESULT_RETENTION_DAYS"))

    @property
    def prune_interval(self) -> float:
        if self._prune_interval is not None:
            return float(self._prune_interval)
        raw = os.environ.get("RESULT_PRUNE_INTERVAL_SECONDS", "").strip()
        if not raw:
            return DEFAULT_PRUNE_INTERVAL
        return float(raw)

    def ensure_writable(self) -> None:
        directory = self.directory
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()

    def start(self) -> None:
        """Prune once now, then keep pruning on a daemon thread."""

        self.ensure_writable()
        removed = self.prune()
        if removed:
            print(f"pruned {removed} expired result(s) from {self.directory}", flush=True)
        with self._thread_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            interval = self.prune_interval
            thread = threading.Thread(target=self._loop, args=(interval,), name="result-prune", daemon=True)
            self._thread = thread
            thread.start()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2)

    def _loop(self, interval: float) -> None:
        while not self._stop.wait(interval):
            try:
                self.prune()
            except Exception as exc:
                print(f"result prune failed: {exc}", flush=True)

    def save_job(self, job) -> Path:
        with job.lock:
            payload = snapshot_payload(job)
        return self.write_payload(payload)

    def write_payload(self, payload: dict) -> Path:
        job_id = str(payload.get("id") or "")
        path = self.path_for(job_id)
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        write_atomic(path, data)
        return path

    def path_for(self, job_id: str) -> Path:
        if not isinstance(job_id, str) or JOB_ID_RE.fullmatch(job_id) is None:
            raise InvalidJobId("invalid job id")
        root = self.directory.resolve()
        path = (root / f"{job_id}.json").resolve()
        if path.parent != root:
            raise InvalidJobId("invalid job id")
        return path

    def get(self, job_id: str) -> dict | None:
        path = self.path_for(job_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            return None
        if not isinstance(payload, dict) or payload.get("id") != job_id:
            return None
        return payload

    def list_results(self) -> list[dict]:
        """Name, date, and duration only. Segment text stays on disk."""

        root = self.directory
        if not root.is_dir():
            return []
        root_resolved = root.resolve()
        rows = []
        for path in root.glob("*.json"):
            if not JOB_ID_RE.fullmatch(path.stem):
                continue
            try:
                if path.resolve().parent != root_resolved:
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeError):
                continue
            if not isinstance(payload, dict):
                continue
            rows.append(
                {
                    "id": path.stem,
                    "name": str(payload.get("name") or payload.get("filename") or "lecture"),
                    "finished_at": _number(payload.get("finished_at")),
                    "duration": _number(payload.get("duration")),
                }
            )
        rows.sort(key=lambda row: (row["finished_at"] or 0, row["id"]), reverse=True)
        return rows

    def delete(self, job_id: str) -> bool:
        path = self.path_for(job_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def prune(self) -> int:
        root = self.directory
        if not root.is_dir():
            return 0
        removed = self._prune_temps(root)
        if self.retention_days == 0:
            return removed
        cutoff = self.clock() - self.retention_days * 86400
        root_resolved = root.resolve()
        for path in list(root.glob("*.json")):
            if not JOB_ID_RE.fullmatch(path.stem):
                continue
            try:
                if path.resolve().parent != root_resolved or not path.is_file():
                    continue
            except OSError:
                continue
            finished = self._finished_at(path)
            if finished is None or finished >= cutoff:
                continue
            try:
                path.unlink()
            except OSError:
                continue
            removed += 1
        return removed

    def _finished_at(self, path: Path) -> float | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            payload = None
        if isinstance(payload, dict) and payload.get("finished_at") is not None:
            number = _number(payload.get("finished_at"))
            if number is not None:
                return number
        try:
            return path.stat().st_mtime
        except OSError:
            return None

    def _prune_temps(self, root: Path) -> int:
        removed = 0
        now = time.time()
        for path in root.glob(".*.tmp"):
            try:
                if now - path.stat().st_mtime < 60:
                    continue
                path.unlink()
                removed += 1
            except OSError:
                continue
        return removed


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number
