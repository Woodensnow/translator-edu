"""Persisted results: atomic write, retention, delete, and path checks."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

from jobs import Job, UploadQueue
from results import InvalidJobId, ResultStore, render_srt, render_txt
from server import GUARD, STORE, Handler, upload_max_body

from test_proxy import _serve


ROOT = Path(__file__).resolve().parents[3]
TOKEN = "a" * 32


def _job(job_id: str, name: str = "第3讲.wav", finished: float = 1_700_000_000.0, duration: float = 95.0) -> Job:
    job = Job(id=job_id, path="", filename=name)
    job.status = "done"
    job.finished_at = finished
    job.total_seconds = duration
    job.segments = [
        {
            "id": "s1",
            "start": 12.4,
            "end": 15.1,
            "text": "Hello",
            "translation": "你好",
            "fallback": False,
            "terms": [],
        }
    ]
    return job


def _fill(job: Job) -> None:
    with job.lock:
        job.total_seconds = 125
        job.segments.append(
            {"id": "s1", "start": 0, "end": 1, "text": "Hi", "translation": "嗨", "fallback": False, "terms": []}
        )


def test_upload_max_body_follows_env(monkeypatch):
    monkeypatch.setenv("UPLOAD_MAX_BODY", "2048")
    assert upload_max_body() == 2048
    monkeypatch.delenv("UPLOAD_MAX_BODY")
    monkeypatch.setenv("ASR_MAX_UPLOAD_MB", "3")
    assert upload_max_body() == 3 * 1024 * 1024
    monkeypatch.delenv("ASR_MAX_UPLOAD_MB")
    assert upload_max_body() == 100 * 1024 * 1024


def test_save_is_atomic_and_roundtrips(tmp_path, monkeypatch):
    calls = []
    real_replace = os.replace

    def spy(src, dst):
        calls.append((str(src), str(dst)))
        assert os.path.basename(src).startswith(".")
        assert str(src).endswith(".tmp")
        real_replace(src, dst)

    monkeypatch.setattr("results.os.replace", spy)
    store = ResultStore(tmp_path, retention_days=7)
    job = _job("abc123abc123")
    path = store.save_job(job)
    assert calls and Path(calls[0][1]) == path
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".*.tmp")) == []
    loaded = store.get(job.id)
    assert loaded["name"] == "第3讲.wav"
    assert loaded["duration"] == 95
    assert loaded["segments"][0]["translation"] == "你好"
    listed = store.list_results()
    assert listed[0]["id"] == job.id
    assert "segments" not in listed[0]
    revived = ResultStore(tmp_path, retention_days=7)
    assert revived.get(job.id)["segments"][0]["text"] == "Hello"


def test_failed_replace_leaves_no_temp_file(tmp_path, monkeypatch):
    def boom(src, dst):
        raise OSError("disk")

    monkeypatch.setattr("results.os.replace", boom)
    directory = tmp_path / "store"
    store = ResultStore(directory, retention_days=7)
    try:
        store.save_job(_job("abc123abc123"))
    except OSError:
        pass
    else:
        raise AssertionError("replace must fail")
    assert list(directory.iterdir()) == []


def test_prune_drops_expired_and_keeps_the_window(tmp_path):
    clock = {"now": 1_700_000_000.0}
    store = ResultStore(tmp_path, retention_days=7, clock=lambda: clock["now"])
    old = _job("a" * 12, finished=clock["now"] - 8 * 86400)
    fresh = _job("b" * 12, name="today.wav", finished=clock["now"] - 6 * 86400)
    edge = _job("c" * 12, finished=clock["now"] - 7 * 86400)
    store.save_job(old)
    store.save_job(fresh)
    store.save_job(edge)
    assert store.prune() == 1
    assert store.get(old.id) is None
    assert store.get(fresh.id)["name"] == "today.wav"
    assert store.get(edge.id) is not None


def test_zero_retention_keeps_files_until_delete(tmp_path):
    clock = {"now": 1_700_000_000.0}
    store = ResultStore(tmp_path, retention_days=0, clock=lambda: clock["now"])
    job = _job("d" * 12, finished=clock["now"] - 400 * 86400)
    store.save_job(job)
    clock["now"] += 10 * 86400
    assert store.prune() == 0
    assert store.get(job.id) is not None
    assert store.delete(job.id) is True
    assert store.get(job.id) is None
    assert store.delete(job.id) is False


def test_start_prunes_immediately_and_again_on_a_timer(tmp_path):
    clock = {"now": 1_700_000_000.0}
    store = ResultStore(tmp_path, retention_days=7, clock=lambda: clock["now"], prune_interval=0.05)
    old = _job("e" * 12, finished=clock["now"] - 30 * 86400)
    fresh = _job("f" * 12, finished=clock["now"])
    store.save_job(old)
    store.save_job(fresh)
    store.start()
    try:
        assert store.get(old.id) is None
        assert store.get(fresh.id) is not None
        clock["now"] += 8 * 86400
        deadline = time.time() + 2
        while time.time() < deadline and store.get(fresh.id) is not None:
            time.sleep(0.05)
        assert store.get(fresh.id) is None
    finally:
        store.close()


def test_invalid_ids_cannot_escape_the_directory(tmp_path):
    store = ResultStore(tmp_path / "results", retention_days=7)
    outside = tmp_path / "secret.txt"
    outside.write_text("keep", encoding="utf-8")
    bad = [
        "../secret",
        "..",
        "abc",
        "ABC123ABC123",
        "a" * 11,
        "g" * 12,
        "/etc/passwd",
        "abc123abc123/../../etc/passwd",
        "..%2F..%2Fetc%2Fpasswd",
        "a" * 12 + ".json",
        "",
        "abcd/ef012345",
    ]
    for job_id in bad:
        try:
            store.path_for(job_id)
        except InvalidJobId:
            pass
        else:
            raise AssertionError(job_id)
    assert outside.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "secret.json").exists()


def test_disk_keeps_results_after_memory_forgets_them(tmp_path):
    store = ResultStore(tmp_path, retention_days=7)
    queue = UploadQueue(_fill, ttl=3600, keep=1, store=store)
    paths = []
    ids = []
    try:
        for name in ("one.wav", "two.wav"):
            handle = tempfile.NamedTemporaryFile(delete=False)
            handle.write(b"x")
            handle.close()
            paths.append(handle.name)
            ids.append(queue.submit(handle.name, name).id)
        queue._queue.join()
        assert queue.get(ids[0]) is None
        assert queue.get(ids[1])["status"] == "done"
        assert {row["id"] for row in store.list_results()} == set(ids)
        for path in paths:
            assert not os.path.exists(path)
    finally:
        for path in paths:
            try:
                os.unlink(path)
            except OSError:
                pass


def test_polling_appends_rows_and_delete_asks_first():
    text = (ROOT / "apps" / "web" / "js" / "app.js").read_text(encoding="utf-8")
    body = text[text.index("async function pollJob"): text.index("function resetCaptions")]
    assert "appendCaptionRows(rowsToAppend" in body
    assert "render()" not in body
    assert "window.confirm" in text
    assert 'method: "DELETE"' in text


def test_txt_and_srt_match_the_page():
    segments = [
        {"start": 12.4, "end": 15.1, "text": "Hello", "translation": "你好"},
        {"start": 1.25, "end": 2, "text": "Backpropagation", "translation": ""},
    ]
    script = (
        "import { toBilingualTxt, toBilingualSrt } from './apps/web/js/captions.js';"
        f"const segments = {json.dumps(segments)};"
        "process.stdout.write(JSON.stringify({txt: toBilingualTxt(segments), srt: toBilingualSrt(segments)}));"
    )
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    expected = json.loads(proc.stdout)
    assert render_txt(segments) == expected["txt"]
    assert render_srt(segments) == expected["srt"]


def test_http_lists_reopens_downloads_and_deletes(monkeypatch):
    monkeypatch.delenv("WEB_ACCESS_TOKEN", raising=False)
    job = _job("abc123abc123")
    STORE.save_job(job)
    web = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    _serve(web)
    port = web.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/results", timeout=3) as response:
            listed = json.loads(response.read().decode())
        assert listed["results"][0]["name"] == "第3讲.wav"
        assert listed["results"][0]["duration"] == 95
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/results/{job.id}", timeout=3) as response:
            opened = json.loads(response.read().decode())
        assert opened["segments"][0]["translation"] == "你好"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/results/{job.id}.txt", timeout=3) as response:
            text = response.read().decode()
            disposition = response.headers.get("Content-Disposition", "")
        assert "Hello" in text and "你好" in text
        assert 'filename="captions.txt"' in disposition
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/results/{job.id}.srt", timeout=3) as response:
            srt = response.read().decode()
        assert "Hello" in srt and "-->" in srt
        request = urllib.request.Request(f"http://127.0.0.1:{port}/results/{job.id}", method="DELETE")
        with urllib.request.urlopen(request, timeout=3) as response:
            assert response.status == 200
        assert STORE.get(job.id) is None
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/results/{job.id}", timeout=3)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("deleted result must 404")
    finally:
        web.shutdown()


def test_http_rejects_path_traversal(monkeypatch):
    monkeypatch.setenv("WEB_ACCESS_TOKEN", TOKEN)
    GUARD.reset()
    outside = Path("/tmp/translator-edu-traversal-secret")
    outside.write_text("keep", encoding="utf-8")
    web = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    _serve(web)
    port = web.server_address[1]
    headers = {"X-Access-Token": TOKEN}
    try:
        for target in (
            "/results/..%2f..%2fetc%2fpasswd",
            "/results/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "/results/abc123abc123/../../etc/passwd",
            "/results/" + "g" * 12,
        ):
            connection = HTTPConnection("127.0.0.1", port, timeout=3)
            connection.request("GET", target, headers=headers)
            response = connection.getresponse()
            response.read()
            assert response.status == 400, target
            connection.close()
        connection = HTTPConnection("127.0.0.1", port, timeout=3)
        connection.request("DELETE", "/results/..%2F..%2Ftmp%2Ftranslator-edu-traversal-secret", headers=headers)
        deleted = connection.getresponse()
        deleted.read()
        assert deleted.status == 400
        assert outside.read_text(encoding="utf-8") == "keep"
        missing = HTTPConnection("127.0.0.1", port, timeout=3)
        missing.request("GET", "/results/" + "a" * 12, headers=headers)
        gone = missing.getresponse()
        gone.read()
        assert gone.status == 404
    finally:
        web.shutdown()
        GUARD.reset()
        outside.unlink(missing_ok=True)


def test_results_require_the_access_token(monkeypatch):
    monkeypatch.setenv("WEB_ACCESS_TOKEN", TOKEN)
    GUARD.reset()
    web = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    _serve(web)
    port = web.server_address[1]
    try:
        bare = HTTPConnection("127.0.0.1", port, timeout=3)
        bare.request("GET", "/results")
        denied = bare.getresponse()
        denied.read()
        assert denied.status == 401
        authed = HTTPConnection("127.0.0.1", port, timeout=3)
        authed.request("GET", "/results", headers={"X-Access-Token": TOKEN})
        opened = authed.getresponse()
        opened.read()
        assert opened.status == 200
    finally:
        web.shutdown()
        GUARD.reset()


def test_result_survives_process_restart(tmp_path):
    port = _free_port()
    env = os.environ.copy()
    env["RESULT_DIR"] = str(tmp_path)
    env["RESULT_RETENTION_DAYS"] = "7"
    env["WEB_ACCESS_TOKEN"] = TOKEN
    env["WEB_REQUIRE_AUTH"] = "1"
    env["WEB_HOST"] = "127.0.0.1"
    env["WEB_PORT"] = str(port)
    env.pop("UPLOAD_MAX_BODY", None)
    root = Path(__file__).resolve().parents[1]
    job = _job("abc123abc123", finished=time.time())
    ResultStore(tmp_path, retention_days=7).save_job(job)

    def start():
        proc = subprocess.Popen(
            [sys.executable, "server.py"],
            cwd=root,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        _wait_health(port, proc)
        return proc

    first = start()
    try:
        with urllib.request.urlopen(
            urllib.request.Request(
                f"http://127.0.0.1:{port}/results/{job.id}",
                headers={"X-Access-Token": TOKEN},
            ),
            timeout=3,
        ) as response:
            assert json.loads(response.read().decode())["segments"][0]["translation"] == "你好"
    finally:
        first.terminate()
        first.wait(timeout=5)
    second = start()
    try:
        with urllib.request.urlopen(
            urllib.request.Request(
                f"http://127.0.0.1:{port}/results/{job.id}.txt",
                headers={"X-Access-Token": TOKEN},
            ),
            timeout=3,
        ) as response:
            body = response.read().decode()
        assert "你好" in body
        assert (tmp_path / f"{job.id}.json").is_file()
    finally:
        second.terminate()
        second.wait(timeout=5)


def test_long_transcript_rss_stays_under_80mb(tmp_path):
    script = r"""
import os
from pathlib import Path
from jobs import Job
from results import ResultStore
segments = []
for i in range(2000):
    segments.append({
        "id": f"s{i}",
        "start": i,
        "end": i + 1,
        "text": "backpropagation " * 8,
        "translation": "反向传播 " * 8,
        "fallback": False,
        "terms": [],
    })
job = Job(id="0123456789ab", path="", filename="lecture.wav")
job.status = "done"
job.finished_at = 1700000000
job.total_seconds = 7200
job.segments = segments
store = ResultStore(os.environ["RESULT_DIR"], retention_days=7)
store.save_job(job)
listed = store.list_results()
assert len(listed) == 1 and "segments" not in listed[0]
loaded = store.get("0123456789ab")
assert len(loaded["segments"]) == 2000
for line in Path(f"/proc/{os.getpid()}/status").read_text().splitlines():
    if line.startswith("VmRSS:"):
        print(line.split()[1])
"""
    env = os.environ.copy()
    env["RESULT_DIR"] = str(tmp_path)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    proc = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    rss_kb = int(proc.stdout.strip().splitlines()[-1])
    # A fresh interpreter plus one saved lecture should stay far under the 128 MB web limit.
    assert rss_kb < 80 * 1024, rss_kb


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _wait_health(port: int, proc: subprocess.Popen) -> None:
    deadline = time.time() + 5
    while time.time() < deadline:
        if proc.poll() is not None:
            err = proc.stderr.read().decode() if proc.stderr else ""
            raise AssertionError(err or "server exited")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.05)
    raise AssertionError("server did not start")


