from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.translator import CTranslate2Translator

ROOT = Path(__file__).resolve().parents[3]
GLOSSARY = ROOT / "glossary"


def test_health_and_glossary_endpoints(client: TestClient):
    health = client.get("/health", headers={"Origin": "http://lecture.local"})
    assert health.status_code == 200
    body = health.json()
    assert body["ready"] is True
    assert body["backend"] == "mock"
    assert body["model"] == "Helsinki-NLP/opus-mt-en-zh"
    assert body["version"] == "0.1.0"
    assert "deep-learning" in body["glossaries"]
    assert health.headers["access-control-allow-origin"] == "*"

    listed = client.get("/glossary")
    assert listed.status_code == 200
    assert listed.json()["default"] == "deep-learning"
    assert listed.json()["columns"] == ["en", "zh", "note"]
    course = next(item for item in listed.json()["glossaries"] if item["name"] == "deep-learning")
    assert course["file"] == "deep-learning.csv"
    assert 60 <= course["terms"] <= 100

    detail = client.get("/glossary/deep-learning")
    assert detail.status_code == 200
    terms = {item["en"]: item["zh"] for item in detail.json()["terms"]}
    assert terms["dropout"] == "暂退法"
    assert detail.json()["columns"] == ["en", "zh", "note"]


def test_translate_text_and_timestamped_segments(client: TestClient):
    text = client.post(
        "/translate",
        json={
            "text": "Backpropagation uses gradient descent to train the neural network.",
            "glossary": "deep-learning",
        },
    )
    assert text.status_code == 200
    body = text.json()
    assert body["source_lang"] == "en"
    assert body["target_lang"] == "zh"
    assert body["backend"] == "mock"
    assert body["translation"] == "反向传播 uses 梯度下降 to train the 神经网络."
    assert body["segments"][0]["id"] == "s0"
    assert body["segments"][0]["start"] is None
    assert [term["en"] for term in body["segments"][0]["terms"]] == [
        "back-propagation",
        "gradient descent",
        "neural network",
    ]
    assert "ZXQ" not in text.text

    segments = client.post(
        "/translate",
        json={
            "text": "ignored dropout",
            "segments": [
                {"id": "cap-1", "start": 12.5, "end": 14, "text": "An epoch ends."},
                {"id": "cap-2", "start": 14, "end": 16.25, "text": "Dropout helps."},
            ],
        },
    )
    assert segments.status_code == 200
    payload = segments.json()
    assert payload["translation"] == "An 迭代周期 ends.\n暂退法 helps."
    assert payload["segments"][0]["id"] == "cap-1"
    assert payload["segments"][0]["start"] == 12.5
    assert payload["segments"][0]["end"] == 14
    assert payload["segments"][0]["source"] == "An epoch ends."
    assert "暂退法" not in payload["segments"][0]["translation"]
    assert payload["segments"][1]["terms"][0]["zh"] == "暂退法"
    assert payload["segments"][1]["terms"][0]["matched"] == "Dropout"


def test_glossary_name_is_case_insensitive(client: TestClient):
    response = client.post("/translate", json={"text": "epoch", "glossary": "Deep-Learning"})
    assert response.status_code == 200
    assert response.json()["glossary"] == "deep-learning"
    assert response.json()["translation"] == "迭代周期"


def test_request_errors(client: TestClient):
    assert client.post("/translate", json={}).status_code == 422
    assert client.post("/translate", json={"text": "hi", "source_lang": "fr"}).status_code == 422
    assert client.post("/translate", json={"text": "hi", "target_lang": "ja"}).status_code == 422
    span = client.post(
        "/translate",
        json={"segments": [{"text": "hi", "start": 5, "end": 1}]},
    )
    assert span.status_code == 422
    missing = client.post("/translate", json={"text": "epoch", "glossary": "not-a-course"})
    assert missing.status_code == 404
    assert "deep-learning" in missing.text

    limited = create_app(Settings(backend="mock", glossary_dir=GLOSSARY, max_chars=8, max_segments=1))
    with TestClient(limited) as local:
        assert local.post("/translate", json={"text": "123456789"}).status_code == 422
        too_many = local.post(
            "/translate",
            json={"segments": [{"text": "a"}, {"text": "b"}]},
        )
        assert too_many.status_code == 422


def test_cpu_backend_reports_not_ready_without_weights(tmp_path: Path):
    application = create_app(
        Settings(
            backend="ctranslate2",
            glossary_dir=GLOSSARY,
            model_dir=tmp_path / "missing-model",
        )
    )
    with TestClient(application) as local:
        health = local.get("/health")
        assert health.status_code == 503
        assert health.json()["ready"] is False
        assert "download_model.py" in health.text
        translated = local.post("/translate", json={"text": "epoch"})
        assert translated.status_code == 503
        assert "download_model.py" in translated.text


def test_blank_text_is_accepted(client: TestClient):
    response = client.post("/translate", json={"text": "   "})
    assert response.status_code == 200
    assert response.json()["translation"] == "   "
    empty = client.post("/translate", json={"text": ""})
    assert empty.status_code == 200
    assert empty.json()["translation"] == ""


def test_default_caption_limit_is_about_two_thousand_characters():
    assert Settings(backend="mock").max_chars == 2000


def test_health_does_not_take_the_inference_lock(tmp_path: Path):
    translator = CTranslate2Translator(
        Settings(backend="ctranslate2", glossary_dir=GLOSSARY, model_dir=tmp_path / "missing-model")
    )
    assert translator._lock.acquire(blocking=False)
    try:
        ready, detail = translator.health()
        translator._translator = object()
        loaded, loaded_detail = translator.health()
    finally:
        translator._lock.release()
    assert ready is False
    assert detail is not None
    assert "download_model.py" in detail
    assert loaded is True
    assert loaded_detail is None


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TRANSLATE_MAX_CHARS", raising=False)
    monkeypatch.setenv("TRANSLATE_BACKEND", "mock")
    monkeypatch.setenv("TRANSLATE_INTRA_THREADS", "2")
    monkeypatch.setenv("TRANSLATE_GLOSSARY_DIR", str(GLOSSARY))
    settings = Settings.from_env()
    assert settings.backend == "mock"
    assert settings.intra_threads == 2
    assert settings.max_chars == 2000
    assert settings.glossary_dir == GLOSSARY

    monkeypatch.setenv("TRANSLATE_BACKEND", "gpu")
    with pytest.raises(ValueError):
        Settings.from_env()
    monkeypatch.setenv("TRANSLATE_BACKEND", "mock")
    monkeypatch.setenv("TRANSLATE_BEAM_SIZE", "nope")
    with pytest.raises(ValueError):
        Settings.from_env()
