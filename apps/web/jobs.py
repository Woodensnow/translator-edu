"""One upload at a time: transcribe a chunk, then translate it, then the next.

Extra jobs stay queued. The page polls and draws bilingual rows as chunks land.
Temp files are removed when the job finishes.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field


BATCH_SEGMENTS = 4
BATCH_CHARS = 1500


@dataclass
class Job:
    id: str
    path: str
    filename: str
    status: str = "queued"
    processed_seconds: float = 0.0
    total_seconds: float | None = None
    segments: list = field(default_factory=list)
    error: str = ""
    code: str = ""
    max_seconds: int = 7200
    lock: threading.Lock = field(default_factory=threading.Lock)

    finished_at: float = 0.0

    def snapshot(self, position: int, after: int = 0) -> dict:
        with self.lock:
            start = max(0, after)
            return {
                "id": self.id,
                "status": self.status,
                "position": position,
                "processed_seconds": self.processed_seconds,
                "total_seconds": self.total_seconds,
                "segments": list(self.segments[start:]),
                "next": len(self.segments),
                "error": self.error,
                "code": self.code,
                "max_seconds": self.max_seconds,
            }


class UploadQueue:
    def __init__(self, runner, ttl: float = 3600.0, keep: int = 8) -> None:
        self._runner = runner
        self._ttl = ttl
        self._keep = keep
        self._jobs: dict[str, Job] = {}
        self._pending: list[str] = []
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="upload-queue", daemon=True)
        self._thread.start()

    def submit(self, path: str, filename: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], path=path, filename=filename)
        with self._lock:
            self._jobs[job.id] = job
            self._pending.append(job.id)
        self._queue.put(job.id)
        return job

    def get(self, job_id: str, after: int = 0) -> dict | None:
        with self._lock:
            self._evict_locked()
            job = self._jobs.get(job_id)
            if job is None:
                return None
            try:
                position = self._pending.index(job_id)
            except ValueError:
                position = 0
        return job.snapshot(position, after)

    def _evict_locked(self) -> None:
        now = time.time()
        finished: list[Job] = []
        for job_id, job in list(self._jobs.items()):
            if job.status not in {"done", "error"}:
                continue
            if job.finished_at and now - job.finished_at > self._ttl:
                self._jobs.pop(job_id, None)
                continue
            finished.append(job)
        finished.sort(key=lambda item: item.finished_at)
        while len(finished) > self._keep:
            old = finished.pop(0)
            self._jobs.pop(old.id, None)

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            with self._lock:
                if job_id in self._pending:
                    self._pending.remove(job_id)
                job = self._jobs.get(job_id)
            if job is None:
                self._queue.task_done()
                continue
            with job.lock:
                job.status = "running"
            try:
                self._runner(job)
                with job.lock:
                    if job.status == "running":
                        job.status = "done"
                    if job.status in {"done", "error"}:
                        job.finished_at = time.time()
            except Exception as exc:
                with job.lock:
                    job.status = "error"
                    job.error = job.error or str(exc) or "处理失败"
                    job.finished_at = time.time()
            finally:
                try:
                    os.unlink(job.path)
                except OSError:
                    pass
                with self._lock:
                    self._evict_locked()
                self._queue.task_done()


def translate_batches(translate_url: str, segments: list[dict], glossary: str, timeout: float) -> list[dict]:
    """Translate a chunk in small batches. A 503 leaves the English text."""

    pending = [dict(segment) for segment in segments]
    output: list[dict] = []
    group: list[dict] = []
    chars = 0
    for segment in pending:
        text = str(segment.get("text") or "")
        if group and (len(group) >= BATCH_SEGMENTS or chars + len(text) > BATCH_CHARS):
            output.extend(_one_batch(translate_url, group, glossary, timeout))
            group = []
            chars = 0
        group.append(segment)
        chars += len(text)
    if group:
        output.extend(_one_batch(translate_url, group, glossary, timeout))
    return output


def _one_batch(translate_url: str, group: list[dict], glossary: str, timeout: float) -> list[dict]:
    body = {
        "glossary": glossary,
        "segments": [
            {
                "id": segment.get("id"),
                "start": segment.get("start") or 0,
                "end": segment.get("end") or 0,
                "text": segment.get("text") or "",
            }
            for segment in group
        ],
    }
    url = translate_url.rstrip("/")
    if not url.endswith("/translate"):
        url += "/translate"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, UnicodeError):
        return [_fallback(segment) for segment in group]
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        return [_fallback(segment) for segment in group]
    by_id = {str(item.get("id")): item for item in payload["segments"] if isinstance(item, dict)}
    rows = []
    for segment in group:
        match = by_id.get(str(segment.get("id")))
        translation = ""
        terms = []
        if match:
            translation = str(match.get("translation") or "").strip()
            terms = match.get("terms") if isinstance(match.get("terms"), list) else []
        if not translation:
            rows.append(_fallback(segment))
            continue
        rows.append({**segment, "translation": translation, "fallback": False, "terms": terms})
    return rows


def _fallback(segment: dict) -> dict:
    return {**segment, "translation": segment.get("text") or "", "fallback": True, "terms": []}


def run_upload(job: Job, path: str, asr_url: str, translate_url: str, glossary: str, timeout: float) -> None:
    """Transcribe one chunk, translate it, then take the next chunk.

    ASR and translate therefore take turns on the two cores instead of running
    their thread pools at the same time.
    """

    import http.client

    parsed = urllib.parse.urlparse(asr_url if "://" in asr_url else "http://" + asr_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    size = os.path.getsize(path)
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    upload_id = None
    try:
        connection.putrequest("POST", "/uploads")
        connection.putheader("Content-Length", str(size))
        connection.putheader("Content-Type", "application/octet-stream")
        connection.putheader("X-Filename", urllib.parse.quote(job.filename or "lecture.bin", safe=""))
        connection.endheaders()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                connection.send(chunk)
        response = connection.getresponse()
        raw = response.read()
        if response.status == 413:
            _mark_too_long(job, _detail(raw))
            return
        if response.status != 200:
            with job.lock:
                job.status = "error"
                job.error = f"Transcription failed ({response.status})"
            return
        meta = json.loads(raw.decode("utf-8"))
        upload_id = meta["id"]
        with job.lock:
            if meta.get("duration") is not None:
                job.total_seconds = float(meta["duration"])
            if meta.get("max_seconds"):
                job.max_seconds = int(meta["max_seconds"])
        while True:
            window = _post_json(host, port, f"/uploads/{upload_id}/next", timeout)
            if window.get("code") == "audio_too_long" or _nested_code(window) == "audio_too_long":
                _mark_too_long(job, window.get("detail") or window)
                return
            translated = translate_batches(translate_url, window.get("segments") or [], glossary, timeout=30)
            with job.lock:
                job.segments.extend(translated)
                job.processed_seconds = float(window.get("processed_seconds") or job.processed_seconds)
                if window.get("total_seconds") is not None:
                    job.total_seconds = float(window["total_seconds"])
            if window.get("done"):
                break
    finally:
        if upload_id:
            try:
                _delete(host, port, f"/uploads/{upload_id}", timeout)
            except OSError:
                pass
        connection.close()


def _detail(raw: bytes):
    try:
        payload = json.loads(raw.decode("utf-8") or "{}")
    except (json.JSONDecodeError, UnicodeError):
        return None
    if isinstance(payload, dict):
        return payload.get("detail", payload)
    return None


def _nested_code(window: dict) -> str:
    detail = window.get("detail")
    if isinstance(detail, dict):
        return str(detail.get("code") or "")
    return str(window.get("code") or "")


def _post_json(host: str, port: int, path: str, timeout: float) -> dict:
    import http.client

    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("POST", path, body=b"{}", headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read()
        payload = json.loads(raw.decode("utf-8") or "{}")
        if response.status == 413 and isinstance(payload, dict):
            return payload
        if response.status != 200:
            raise RuntimeError(f"Transcription failed ({response.status})")
        return payload if isinstance(payload, dict) else {}
    finally:
        connection.close()


def _delete(host: str, port: int, path: str, timeout: float) -> None:
    import http.client

    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("DELETE", path)
        connection.getresponse().read()
    finally:
        connection.close()


def _mark_too_long(job: Job, detail) -> None:
    max_seconds = 7200
    if isinstance(detail, dict) and detail.get("max_seconds"):
        max_seconds = int(detail["max_seconds"])
    with job.lock:
        job.status = "error"
        job.code = "audio_too_long"
        job.max_seconds = max_seconds
        job.error = "录音太长，请分段上传"
        job.segments = []
