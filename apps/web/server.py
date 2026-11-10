#!/usr/bin/env python3
"""Public captions page for one 2-core host.

The browser only talks to this process. ASR and translate stay on the compose
network. Static files are an allowlist: this module is not served.

v1 is plain HTTP. The access token is an HttpOnly SameSite=Strict cookie
without Secure. TODO(https): add Secure when Caddy terminates TLS.
"""

from __future__ import annotations

import json
import os
import select
import socket
import ssl
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from auth import (
    AuthGuard,
    clear_cookie_header,
    configured_token,
    ensure_token,
    presented_token,
    set_cookie_header,
    tokens_equal,
)
from jobs import Job

ROOT = Path(__file__).resolve().parent
TRANSLATE_MAX_BODY = 16 * 1024
TRANSCRIBE_MAX_BODY = 100 * 1024 * 1024
GUARD = AuthGuard()

STATIC_FILES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/login": "login.html",
    "/css/styles.css": "css/styles.css",
    "/js/app.js": "js/app.js",
    "/js/audio-capture.js": "js/audio-capture.js",
    "/js/captions.js": "js/captions.js",
    "/js/pcm-processor.js": "js/pcm-processor.js",
    "/js/translate-client.js": "js/translate-client.js",
}


class ContentLengthError(ValueError):
    def __init__(self, status: int, message: str) -> None:
        self.status = status
        super().__init__(message)


def parse_content_length(raw: str | None, limit: int) -> int:
    """Reject a missing, non-integer, or negative length before any read.

    ``rfile.read(-1)`` would read until the socket closes, so a negative
    Content-Length must never reach the reader.
    """

    if raw is None or str(raw).strip() == "":
        raise ContentLengthError(400, "Content-Length is required.")
    try:
        length = int(str(raw).strip())
    except ValueError as exc:
        raise ContentLengthError(400, "Content-Length is invalid.") from exc
    if length < 0:
        raise ContentLengthError(400, "Content-Length is invalid.")
    if length > limit:
        raise ContentLengthError(413, "Request is too large.")
    return length


def config() -> dict:
    return public_config()


def public_config() -> dict:
    """Fields the browser needs. Internal service URLs stay on the server."""

    return {
        "course": os.environ.get("COURSE_ID", "deep-learning"),
        "courseTitle": os.environ.get("COURSE_TITLE", "Deep Learning"),
        "jobsPath": "/jobs",
        "translatePath": "/translate",
        "asrWsPath": "/ws/transcribe",
        "maxAudioSeconds": int(float(os.environ.get("ASR_MAX_AUDIO_SECONDS", "7200"))),
    }


def translate_upstream() -> str:
    base = os.environ.get("TRANSLATE_URL", "http://translate:8080").strip().rstrip("/")
    if base.endswith("/translate"):
        return base
    return base + "/translate"


def transcribe_upstream() -> str:
    return os.environ.get("ASR_HTTP_URL", "http://asr:8001").strip().rstrip("/")


def asr_ws_upstream() -> str:
    return os.environ.get("ASR_WS_URL", "ws://asr:8001/ws/transcribe").strip()


def translate_timeout() -> float:
    return float(os.environ.get("TRANSLATE_TIMEOUT_SECONDS", "8"))


def forward_http(url: str, body: bytes, content_type: str | None, timeout: float) -> tuple[int, bytes, str]:
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Accept", "application/json")
    if content_type:
        request.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
            kind = response.headers.get("Content-Type", "application/json")
            return response.status, payload, kind
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        kind = exc.headers.get("Content-Type", "application/json") if exc.headers else "application/json"
        return exc.code, payload, kind
    except TimeoutError:
        return 504, b'{"detail":"upstream timed out"}', "application/json"
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            return 504, b'{"detail":"upstream timed out"}', "application/json"
        return 503, b'{"detail":"upstream unavailable"}', "application/json"


def proxy_websocket(client: socket.socket, upstream_url: str, client_headers) -> None:
    parsed = urllib.parse.urlparse(upstream_url)
    if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
        _send_simple(client, 500, b'{"detail":"ASR websocket URL is invalid"}')
        return
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    path = parsed.path or "/ws/transcribe"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    try:
        upstream = socket.create_connection((parsed.hostname, port), timeout=5)
    except OSError:
        _send_simple(client, 503, b'{"detail":"ASR is unavailable"}')
        return
    try:
        if parsed.scheme == "wss":
            context = ssl.create_default_context()
            upstream = context.wrap_socket(upstream, server_hostname=parsed.hostname)
        host_header = parsed.hostname if parsed.port is None else f"{parsed.hostname}:{port}"
        lines = [f"GET {path} HTTP/1.1", f"Host: {host_header}"]
        for name in (
            "Upgrade",
            "Connection",
            "Sec-WebSocket-Key",
            "Sec-WebSocket-Version",
            "Sec-WebSocket-Protocol",
            "Sec-WebSocket-Extensions",
            "Cookie",
        ):
            value = client_headers.get(name)
            if value and name != "Cookie":
                lines.append(f"{name}: {value}")
        upstream.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("ascii"))
        head = _read_headers(upstream)
        client.sendall(head)
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            return
        upstream.settimeout(None)
        client.settimeout(None)
        _shuttle(client, upstream)
    finally:
        try:
            upstream.close()
        except OSError:
            pass


def _read_headers(sock: socket.socket) -> bytes:
    buffer = b""
    while b"\r\n\r\n" not in buffer:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buffer += chunk
        if len(buffer) > 65536:
            break
    return buffer


def _shuttle(left: socket.socket, right: socket.socket) -> None:
    pair = [left, right]
    try:
        while True:
            readable, _, _ = select.select(pair, [], [])
            for sock in readable:
                data = sock.recv(65536)
                if not data:
                    return
                other = right if sock is left else left
                other.sendall(data)
    except OSError:
        return


def _send_simple(sock: socket.socket, status: int, body: bytes) -> None:
    reason = {500: "Server Error", 503: "Service Unavailable"}.get(status, "Error")
    head = (
        f"HTTP/1.1 {status} {reason}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    )
    try:
        sock.sendall(head.encode("ascii") + body)
    except OSError:
        pass


def static_path(url_path: str) -> Path | None:
    rel = STATIC_FILES.get(url_path)
    if rel is None:
        return None
    path = (ROOT / rel).resolve()
    root = ROOT.resolve()
    if path != root and root not in path.parents:
        return None
    if not path.is_file():
        return None
    return path


def login_html(wrong: bool) -> bytes:
    path = ROOT / "login.html"
    text = path.read_text(encoding="utf-8")
    message = '<p class="wrong" role="alert">口令不对</p>' if wrong else ""
    return text.replace("<!-- message -->", message).encode("utf-8")


def _runner(job: Job) -> None:
    from jobs import run_upload

    run_upload(
        job,
        job.path,
        transcribe_upstream(),
        os.environ.get("TRANSLATE_URL", "http://translate:8080"),
        os.environ.get("COURSE_ID", "deep-learning"),
        float(os.environ.get("ASR_UPLOAD_TIMEOUT_SECONDS", "600")),
    )


def _build_queue():
    from jobs import UploadQueue

    return UploadQueue(_runner)


QUEUE = _build_queue()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send_bytes(200, b'{"status":"ok"}', "application/json")
            return
        if path == "/login":
            self._send_bytes(200, login_html(False), "text/html; charset=utf-8", extra=[clear_cookie_header()])
            return
        if path == "/ws/transcribe" and "websocket" in self.headers.get("Upgrade", "").lower():
            if not self._allow_api():
                return
            self.close_connection = True
            proxy_websocket(self.connection, asr_ws_upstream(), self.headers)
            return
        if path == "/config.json":
            if not self._allow_api():
                return
            self._send_bytes(200, json.dumps(public_config()).encode("utf-8"), "application/json; charset=utf-8")
            return
        if path.startswith("/jobs/"):
            if not self._allow_api():
                return
            job_id = path.removeprefix("/jobs/")
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                after = int((query.get("after") or ["0"])[0])
            except ValueError:
                after = 0
            snapshot = QUEUE.get(job_id, after=max(0, after))
            if snapshot is None:
                self._send_bytes(404, b'{"detail":"Job was not found."}', "application/json")
                return
            self._send_bytes(200, json.dumps(snapshot).encode("utf-8"), "application/json")
            return
        file_path = static_path(path)
        if file_path is None:
            self.send_error(404)
            return
        body = file_path.read_bytes()
        self._send_bytes(200, body, _mime(file_path))

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/login":
            self._login()
            return
        if path == "/jobs":
            if not self._allow_api():
                return
            self._enqueue()
            return
        if path == "/translate":
            if not self._allow_api():
                return
            self._proxy_post(translate_upstream(), TRANSLATE_MAX_BODY, translate_timeout())
            return
        self.send_error(404)

    def _allow_api(self) -> bool:
        """API calls are never locked out. A valid cookie still works."""

        expected = configured_token()
        if not expected:
            return True
        presented = presented_token(self.headers)
        if presented and tokens_equal(presented, expected):
            return True
        self._send_bytes(401, b'{"detail":"unauthorized"}', "application/json")
        return False

    def _login(self) -> None:
        expected = configured_token()
        try:
            length = parse_content_length(self.headers.get("Content-Length"), 4096)
        except ContentLengthError as exc:
            self._send_bytes(exc.status, json.dumps({"detail": str(exc)}).encode(), "application/json")
            return
        raw = self.rfile.read(length) if length else b""
        form = urllib.parse.parse_qs(raw.decode("utf-8", errors="replace"))
        token = (form.get("token") or [""])[0]
        if not expected:
            self._redirect("/")
            return
        state = GUARD.check(self.client_address[0], token, expected)
        if state == "locked":
            retry = GUARD.retry_after(self.client_address[0])
            self._send_bytes(429, "口令尝试过多，请稍后再试".encode(), "text/plain; charset=utf-8", extra=[f"Retry-After: {retry}"])
            return
        if state != "ok":
            self._send_bytes(401, login_html(True), "text/html; charset=utf-8")
            return
        self._redirect("/", extra=[set_cookie_header(expected)])

    def _enqueue(self) -> None:
        try:
            length = parse_content_length(self.headers.get("Content-Length"), TRANSCRIBE_MAX_BODY)
        except ContentLengthError as exc:
            self._send_bytes(exc.status, json.dumps({"detail": str(exc)}).encode(), "application/json")
            return
        handle = tempfile.NamedTemporaryFile(prefix="caption-job-", suffix=".upload", delete=False)
        path = handle.name
        remaining = length
        try:
            while remaining:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                handle.write(chunk)
                remaining -= len(chunk)
        except Exception:
            handle.close()
            try:
                os.unlink(path)
            except OSError:
                pass
            raise
        handle.close()
        if remaining:
            try:
                os.unlink(path)
            except OSError:
                pass
            self._send_bytes(400, b'{"detail":"Upload ended early."}', "application/json")
            return
        filename = _filename_header(self.headers.get("X-Filename"))
        job = QUEUE.submit(path, filename)
        body = json.dumps(QUEUE.get(job.id) or {}).encode()
        self._send_bytes(202, body, "application/json")

    def _proxy_post(self, url: str, limit: int, timeout: float) -> None:
        try:
            length = parse_content_length(self.headers.get("Content-Length"), limit)
        except ContentLengthError as exc:
            self._send_bytes(exc.status, json.dumps({"detail": str(exc)}).encode(), "application/json")
            return
        body = self.rfile.read(length) if length else b""
        status, payload, content_type = forward_http(url, body, self.headers.get("Content-Type"), timeout)
        self._send_bytes(status, payload, content_type or "application/octet-stream")

    def _redirect(self, location: str, extra: list[str] | None = None) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        for item in extra or []:
            name, _, value = item.partition(": ")
            if name == "Set-Cookie" or item.startswith("web_access=") or "HttpOnly" in item:
                self.send_header("Set-Cookie", item if "HttpOnly" in item else value)
            else:
                self.send_header(name or "Set-Cookie", value or item)
        self.end_headers()

    def _send_bytes(self, status: int, body: bytes, content_type: str, extra: list[str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        for item in extra or []:
            if item.startswith("Retry-After"):
                self.send_header("Retry-After", item.split(":", 1)[1].strip())
            else:
                self.send_header("Set-Cookie", item)
        self.end_headers()
        self.wfile.write(body)


def _filename_header(raw: str | None) -> str:
    """Decode an RFC 3986 percent-encoded X-Filename. The header itself is ASCII."""

    name = urllib.parse.unquote((raw or "").strip(), encoding="utf-8", errors="replace")
    name = Path(name).name
    if not name or name in {".", ".."}:
        return "lecture.bin"
    return name


def _mime(path: Path) -> str:
    name = path.name
    if name.endswith(".css"):
        return "text/css; charset=utf-8"
    if name.endswith(".js"):
        return "text/javascript; charset=utf-8"
    return "text/html; charset=utf-8"


def main() -> None:
    env = {key: value for key, value in os.environ.items()}
    ensure_token(env)
    if not configured_token():
        print("WARNING: WEB_ACCESS_TOKEN is unset; the captions API is open.", flush=True)
    host = os.environ.get("WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("WEB_PORT", "8080"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"captions page on http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
