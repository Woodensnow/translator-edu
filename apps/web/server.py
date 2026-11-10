#!/usr/bin/env python3
"""Public captions page.

The browser only talks to this process. ASR and translate stay on the private
compose network. This server forwards:

- WebSocket /ws/transcribe  -> ASR_WS_URL (default ws://asr:8001/ws/transcribe)
- POST /transcribe          -> ASR_HTTP_URL/transcribe
- POST /translate           -> TRANSLATE_URL/translate

The translate body is not rewritten. See docs/translate-contract.md.
"""

from __future__ import annotations

import json
import os
import select
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRANSLATE_MAX_BODY = 16 * 1024
TRANSCRIBE_MAX_BODY = 100 * 1024 * 1024


def config() -> dict:
    return {
        "course": os.environ.get("COURSE_ID", "deep-learning"),
        "courseTitle": os.environ.get("COURSE_TITLE", "Deep Learning"),
        "asrWsPath": "/ws/transcribe",
        "transcribePath": "/transcribe",
        "translatePath": "/translate",
        "translateUrl": os.environ.get("TRANSLATE_URL", "http://translate:8080"),
        "asrHttpUrl": os.environ.get("ASR_HTTP_URL", "http://asr:8001"),
    }


def translate_upstream() -> str:
    base = os.environ.get("TRANSLATE_URL", "http://translate:8080").strip().rstrip("/")
    if base.endswith("/translate"):
        return base
    return base + "/translate"


def transcribe_upstream() -> str:
    return os.environ.get("ASR_HTTP_URL", "http://asr:8001").strip().rstrip("/") + "/transcribe"


def asr_ws_upstream() -> str:
    return os.environ.get("ASR_WS_URL", "ws://asr:8001/ws/transcribe").strip()


def translate_timeout() -> float:
    return float(os.environ.get("TRANSLATE_TIMEOUT_SECONDS", "8"))


def forward_http(url: str, body: bytes, content_type: str | None, timeout: float) -> tuple[int, bytes, str]:
    """POST a body upstream and return status, payload, and content type.

    HTTP errors from the upstream are returned as-is so a translate 503 still
    looks like a 503 to the browser. Connection failures become 503. A timeout
    becomes 504, which the page treats like a translate timeout.
    """

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
    """Tunnel one WebSocket, including the handshake, to the ASR service."""

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
        ):
            value = client_headers.get(name)
            if value:
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


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        return

    def list_directory(self, path):
        self.send_error(404)
        return None

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/ws/transcribe" and "websocket" in self.headers.get("Upgrade", "").lower():
            self.close_connection = True
            proxy_websocket(self.connection, asr_ws_upstream(), self.headers)
            return
        if path == "/config.json":
            self._send_bytes(200, json.dumps(config()).encode("utf-8"), "application/json; charset=utf-8")
            return
        if path == "/health":
            self._send_bytes(200, b'{"status":"ok"}', "application/json")
            return
        super().do_GET()

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/translate":
            self._proxy_post(translate_upstream(), TRANSLATE_MAX_BODY, translate_timeout())
            return
        if path == "/transcribe":
            timeout = float(os.environ.get("ASR_UPLOAD_TIMEOUT_SECONDS", "180"))
            self._proxy_post(transcribe_upstream(), TRANSCRIBE_MAX_BODY, timeout)
            return
        self.send_error(404)

    def _proxy_post(self, url: str, limit: int, timeout: float) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "Content-Length is invalid")
            return
        if length > limit:
            self.send_error(413, "Request is too large")
            return
        body = self.rfile.read(length) if length else b""
        status, payload, content_type = forward_http(url, body, self.headers.get("Content-Type"), timeout)
        self._send_bytes(status, payload, content_type or "application/octet-stream")

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    host = os.environ.get("WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("WEB_PORT", "8080"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"captions page on http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
