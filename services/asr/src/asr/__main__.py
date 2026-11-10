"""Run the ASR HTTP and WebSocket server."""

from __future__ import annotations

import os

import uvicorn

from asr.server import create_app


def main() -> None:
    host = os.environ.get("ASR_HOST", "0.0.0.0")
    port = int(os.environ.get("ASR_PORT", "8001"))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
