"""Uploads wait their turn, and rows appear before the job is done."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

from server import QUEUE, Handler

from test_proxy import _serve


def test_one_job_runs_at_a_time_and_segments_arrive_early(monkeypatch):
    monkeypatch.delenv("WEB_ACCESS_TOKEN", raising=False)
    started = []
    release = threading.Event()

    def runner(job):
        started.append(job.id)
        with job.lock:
            job.total_seconds = 120
            job.processed_seconds = 60
            job.segments.append(
                {"id": f"{job.id}-seg-1", "start": 0, "end": 60, "text": "first half", "translation": "前半", "fallback": False}
            )
        release.wait(2)
        with job.lock:
            job.processed_seconds = 120
            job.segments.append(
                {"id": f"{job.id}-seg-2", "start": 60, "end": 120, "text": "second half", "translation": "后半", "fallback": False}
            )

    previous = QUEUE._runner
    QUEUE._runner = runner
    web = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    _serve(web)
    port = web.server_address[1]
    paths = []
    try:
        ids = []
        for _ in range(2):
            handle = tempfile.NamedTemporaryFile(delete=False)
            handle.write(b"audio")
            handle.close()
            paths.append(handle.name)
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/jobs",
                data=b"audio",
                method="POST",
                headers={"Content-Type": "application/octet-stream", "Content-Length": "5"},
            )
            with urllib.request.urlopen(request, timeout=3) as response:
                assert response.status == 202
                ids.append(json.loads(response.read().decode())["id"])
        seen_partial = False
        deadline = time.time() + 3
        while time.time() < deadline and not seen_partial:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/jobs/{ids[0]}", timeout=3) as response:
                body = json.loads(response.read().decode())
            if body["segments"] and body["status"] == "running":
                seen_partial = True
                assert body["segments"][0]["translation"] == "前半"
        assert seen_partial
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/jobs/{ids[1]}", timeout=3) as response:
            queued = json.loads(response.read().decode())
        assert queued["status"] == "queued"
        release.set()
        deadline = time.time() + 3
        done = {}
        while time.time() < deadline:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/jobs/{ids[1]}", timeout=3) as response:
                done = json.loads(response.read().decode())
            if done["status"] == "done":
                break
            time.sleep(0.05)
        assert done["status"] == "done"
        assert len(done["segments"]) == 2
        assert started == ids
    finally:
        release.set()
        QUEUE._runner = previous
        web.shutdown()
        for path in paths:
            try:
                os.unlink(path)
            except OSError:
                pass
