#!/usr/bin/env python3
"""Download the Whisper weights used by the ASR service.

Weights are stored in ``ASR_MODEL_DIR`` (or the Hugging Face cache when that
variable is unset). They are not committed. The server also downloads on the
first transcription if this script was not run.

Example, on the transcription host::

    ASR_MODEL=base.en ASR_MODEL_DIR=/var/lib/translator-edu/models \\
        python scripts/download_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr.engine import WhisperEngine  # noqa: E402
from asr.settings import Settings  # noqa: E402


def main() -> None:
    settings = Settings.from_env()
    print(
        f"Downloading {settings.model} ({settings.compute_type}) "
        f"into {settings.model_dir or 'the default Hugging Face cache'}",
        flush=True,
    )
    WhisperEngine(settings).ensure_loaded()
    print("ready", flush=True)


if __name__ == "__main__":
    main()
