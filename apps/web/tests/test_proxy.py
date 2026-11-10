"""The public page forwards translate and ASR without rewriting the contract."""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from server import Handler, config, translate_upstream


ANNA = {
    "source_lang": "en",
    "target_lang": "zh",
    "glossary": "deep-learning",
    "model": "Helsinki-NLP/opus-mt-en-zh",
    "backend": "mock",
    "translation": "反向传播使用梯度下降。",
    "segments": [
        {
            "id": "seg-1",
            "start": 12.4,
            "end": 15.1,
            "source": "Backpropagation uses gradient descent.",
            "translation": "反向传播使用梯度下降。",
            "terms": [
                {"en": "backpropagation", "zh": "反向传播", "note": "", "matched": "Backpropagation"}
            ],
        }
    ],
}


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        self.server.seen = json.loads(raw.decode())
        if self.server.mode == "unavailable":
            body = b'{"detail":"weights are not converted yet"}'
            self.send_response(503)
        else:
            body = json.dumps(ANNA).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _post(port, body):
    data = json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/translate",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def test_config_hides_internal_urls(monkeypatch):
    monkeypatch.delenv("COURSE_ID", raising=False)
    monkeypatch.setenv("TRANSLATE_URL", "http://translate:8080")
    monkeypatch.setenv("ASR_HTTP_URL", "http://asr:8001")
    body = config()
    assert body["course"] == "deep-learning"
    assert body["translatePath"] == "/translate"
    assert body["maxAudioSeconds"] == 7200
    blob = json.dumps(body)
    assert "translate:8080" not in blob
    assert "asr:8001" not in blob
    assert translate_upstream() == "http://translate:8080/translate"


def test_translate_proxy_forwards_segments_and_503(monkeypatch):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream.mode = "ok"
    upstream.seen = None
    _serve(upstream)
    monkeypatch.setenv("TRANSLATE_URL", f"http://127.0.0.1:{upstream.server_address[1]}")
    web = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    _serve(web)
    port = web.server_address[1]
    request_body = {
        "glossary": "deep-learning",
        "segments": [{"id": "seg-1", "start": 12.4, "end": 15.1, "text": "Backpropagation uses gradient descent."}],
    }
    try:
        status, payload = _post(port, request_body)
        assert status == 200
        assert upstream.seen == request_body
        assert payload["segments"][0]["id"] == "seg-1"
        assert payload["segments"][0]["translation"] == "反向传播使用梯度下降。"
        assert payload["translation"]

        upstream.mode = "unavailable"
        status, payload = _post(port, request_body)
        assert status == 503
        assert "detail" in payload
    finally:
        web.shutdown()
        upstream.shutdown()


def test_index_offers_both_columns():
    web = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    _serve(web)
    port = web.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as response:
            html = response.read().decode()
        assert response.status == 200
        assert 'id="captions"' in html
        assert 'id="live-btn"' in html
        assert 'id="history-list"' in html
        assert "过去的结果" in html
        assert "中文" in html
    finally:
        web.shutdown()


def test_websocket_proxy_reaches_asr(monkeypatch):
    received = bytearray()
    ready = threading.Event()

    def asr():
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port_holder.append(listener.getsockname()[1])
        ready.set()
        conn, _addr = listener.accept()
        data = b""
        while b"\r\n\r\n" not in data:
            data += conn.recv(4096)
        received.extend(data)
        conn.sendall(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n")
        conn.sendall(b"frame")
        conn.close()
        listener.close()

    port_holder = []
    threading.Thread(target=asr, daemon=True).start()
    assert ready.wait(2)
    monkeypatch.setenv("ASR_WS_URL", f"ws://127.0.0.1:{port_holder[0]}/ws/transcribe")
    web = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    _serve(web)
    try:
        client = socket.create_connection(("127.0.0.1", web.server_address[1]), timeout=3)
        client.sendall(
            b"GET /ws/transcribe HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Upgrade: websocket\r\n"
            b"Connection: Upgrade\r\n"
            b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
            b"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        data = b""
        client.settimeout(3)
        while b"frame" not in data:
            chunk = client.recv(4096)
            if not chunk:
                break
            data += chunk
        client.close()
    finally:
        web.shutdown()
    assert b" 101 " in data
    assert data.endswith(b"frame")
    assert b"/ws/transcribe" in received
    assert b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" in received
