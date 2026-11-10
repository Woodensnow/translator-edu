import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
os.environ["TRANSLATE_BACKEND"] = "mock"
os.environ["TRANSLATE_GLOSSARY_DIR"] = str(ROOT / "glossary")

from app.config import Settings
from app.main import create_app


@pytest.fixture
def client():
    application = create_app(Settings(backend="mock", glossary_dir=ROOT / "glossary"))
    with TestClient(application) as test_client:
        yield test_client
