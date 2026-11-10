"""A real uvicorn process must accept a WebSocket upgrade.

Starlette's TestClient does not use the websockets or wsproto package, so it
stays green when the image has plain ``uvicorn`` and live captions 404.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
import urllib.request

import pytest
import uvicorn

from asr.server import create_app
from support import FakeEngine
from test_server import _settings


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_uvicorn_accepts_a_websocket_handshake():
    pytest.importorskip("websockets")
    import websockets

    app = create_app(settings=_settings(), engine=FakeEngine())
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="uvicorn-ws", daemon=True)
    thread.start()
    deadline = time.time() + 10
    try:
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(0.05)
        else:
            raise AssertionError("uvicorn did not start")

        async def talk():
            async with websockets.connect(f"ws://127.0.0.1:{port}/ws/transcribe") as ws:
                await ws.send(json.dumps({"type": "start", "sample_rate": 16000}))
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                return json.loads(raw)

        ready = asyncio.run(talk())
    finally:
        server.should_exit = True
        thread.join(timeout=5)
    assert ready["type"] == "ready"
    assert ready["encoding"] == "pcm_s16le"
