"""Keep saved-result files inside the test tmp directory."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def result_dir(tmp_path, monkeypatch):
    path = tmp_path / "results"
    path.mkdir()
    monkeypatch.setenv("RESULT_DIR", str(path))
    monkeypatch.setenv("RESULT_RETENTION_DAYS", "7")
    return path
