"""Negative Content-Length is rejected, and source files are not served."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import urllib.parse
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

from server import QUEUE, Handler, parse_content_length, ContentLengthError

from test_proxy import _serve


def test_parse_content_length_rejects_missing_negative_and_junk():
    for raw in (None, "", "nope", "-1", "-5"):
        try:
            parse_content_length(raw, 100)
        except ContentLengthError as exc:
            assert exc.status == 400
        else:
            raise AssertionError(raw)
    try:
        parse_content_length("101", 100)
    except ContentLengthError as exc:
        assert exc.status == 413
    else:
        raise AssertionError("expected 413")
    assert parse_content_length("0", 100) == 0


def test_negative_content_length_does_not_read_the_body(monkeypatch):
    monkeypatch.delenv("WEB_ACCESS_TOKEN", raising=False)
    web = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    _serve(web)
    port = web.server_address[1]
    try:
        connection = HTTPConnection("127.0.0.1", port, timeout=3)
        connection.putrequest("POST", "/translate")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", "-1")
        connection.endheaders()
        response = connection.getresponse()
        body = response.read()
        assert response.status == 400
        assert b"Content-Length" in body

        missing = HTTPConnection("127.0.0.1", port, timeout=3)
        missing.putrequest("POST", "/translate")
        missing.putheader("Content-Type", "application/json")
        missing.endheaders()
        missed = missing.getresponse()
        assert missed.status == 400
        missed.read()

        source = HTTPConnection("127.0.0.1", port, timeout=3)
        source.request("GET", "/server.py")
        hidden = source.getresponse()
        hidden.read()
        assert hidden.status == 404
    finally:
        web.shutdown()


def test_chinese_filename_is_percent_encoded(monkeypatch):
    monkeypatch.delenv("WEB_ACCESS_TOKEN", raising=False)
    seen = []

    def runner(job):
        seen.append(job.filename)

    previous = QUEUE._runner
    QUEUE._runner = runner
    web = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    _serve(web)
    port = web.server_address[1]
    try:
        name = "第3讲.wav"
        encoded = urllib.parse.quote(name, safe="")
        encoded.encode("latin-1")
        body = b"not-audio"
        connection = HTTPConnection("127.0.0.1", port, timeout=3)
        connection.request(
            "POST",
            "/jobs",
            body=body,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(body)),
                "X-Filename": encoded,
            },
        )
        response = connection.getresponse()
        response.read()
        assert response.status == 202
        deadline = time.time() + 2
        while time.time() < deadline and not seen:
            time.sleep(0.02)
        assert seen == [name]
    finally:
        QUEUE._runner = previous
        web.shutdown()


def test_large_upload_keeps_web_rss_under_80mb():
    env = os.environ.copy()
    env.pop("WEB_ACCESS_TOKEN", None)
    env["WEB_REQUIRE_AUTH"] = "0"
    env["WEB_HOST"] = "127.0.0.1"
    env["WEB_PORT"] = "18771"
    env["ASR_HTTP_URL"] = "http://127.0.0.1:9"
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.Popen(
        [sys.executable, "server.py"],
        cwd=root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    peaks = []

    def sample():
        while proc.poll() is None:
            try:
                text = Path(f"/proc/{proc.pid}/status").read_text()
            except OSError:
                return
            for line in text.splitlines():
                if line.startswith("VmRSS:"):
                    peaks.append(int(line.split()[1]))
            time.sleep(0.02)

    thread = threading.Thread(target=sample, daemon=True)
    try:
        for _ in range(50):
            if proc.poll() is not None:
                raise AssertionError(proc.stderr.read().decode() if proc.stderr else "server exited")
            try:
                probe = HTTPConnection("127.0.0.1", 18771, timeout=1)
                probe.request("GET", "/health")
                probed = probe.getresponse()
                probed.read()
                probe.close()
                if probed.status == 200:
                    break
            except Exception:
                time.sleep(0.05)
        else:
            raise AssertionError("web server did not start")
        thread.start()
        connection = HTTPConnection("127.0.0.1", 18771, timeout=30)
        payload = b"\0" * (32 * 1024 * 1024)
        connection.request(
            "POST",
            "/jobs",
            body=payload,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(payload)),
                "X-Filename": "lecture.wav",
            },
        )
        response = connection.getresponse()
        response.read()
        assert response.status == 202
        time.sleep(0.2)
        assert peaks, "no RSS samples"
        assert max(peaks) < 80 * 1024, max(peaks)
    finally:
        proc.terminate()
        proc.wait(timeout=5)
