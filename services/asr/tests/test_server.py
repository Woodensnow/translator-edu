"""HTTP and WebSocket API with a fake engine. No model download."""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from asr.engine import WhisperEngine
from asr.glossary import GlossaryFormatError
from asr.server import create_app
from asr.settings import Settings
from support import FIXTURE, BoomEngine, FakeEngine, sine_wav


def _settings(**overrides) -> Settings:
    base = Settings.from_env(
        {
            "ASR_GLOSSARY_PATH": str(FIXTURE),
            "ASR_WINDOW_SECONDS": "1",
            "ASR_PARTIAL_SECONDS": "0",
        }
    )
    return replace(base, **overrides) if overrides else base


def _messages(ws, limit=20):
    found = []
    for _ in range(limit):
        message = ws.receive_json()
        found.append(message)
        if message.get("type") in {"done", "error"}:
            break
    return found


def test_health_reports_fixture_terms_without_loading_weights():
    engine = WhisperEngine(Settings.from_env({}))
    app = create_app(settings=_settings(), engine=engine)
    with TestClient(app) as client:
        response = client.get("/health", headers={"Origin": "http://localhost:8080"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "base.en"
    assert body["compute_type"] == "int8"
    assert body["cpu_threads"] == 2
    assert body["glossary_terms"] == 3
    assert body["glossary_path"].endswith("deep-learning.csv")
    assert body["model_loaded"] is False
    assert engine.loaded is False
    assert response.headers["access-control-allow-origin"] == "*"


def test_batch_passes_english_terms_and_not_notes():
    engine = FakeEngine()
    app = create_app(settings=_settings(), engine=engine)
    with TestClient(app) as client:
        response = client.post(
            "/transcribe",
            files={"file": ("lecture.wav", sine_wav(), "audio/wav")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["language"] == "en"
    assert body["segments"][0]["id"].endswith("-seg-1")
    assert body["segments"][0]["id"] != "seg-1"
    assert body["segments"][0]["text"] == engine.text
    assert body["text"] == engine.text
    call = engine.calls[0]
    assert call["vad"] is True
    for term in ("backpropagation", "gradient descent", "activation function"):
        assert term in call["initial_prompt"]
        assert term in call["hotwords"]
    for excluded in ("反向传播", "ReLU", "sigmoid"):
        assert excluded not in call["initial_prompt"]
        assert excluded not in call["hotwords"]


def test_batch_rejects_empty_unknown_and_oversize_and_hides_engine_errors():
    app = create_app(settings=_settings(), engine=FakeEngine())
    with TestClient(app) as client:
        empty = client.post("/transcribe", files={"file": ("lecture.wav", b"", "audio/wav")})
        unknown = client.post("/transcribe", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert empty.status_code == 400
    assert unknown.status_code == 400

    tiny = create_app(settings=_settings(max_upload_bytes=8), engine=FakeEngine())
    with TestClient(tiny) as client:
        oversize = client.post("/transcribe", files={"file": ("lecture.wav", b"012345678", "audio/wav")})
    assert oversize.status_code == 413

    broken = create_app(settings=_settings(), engine=BoomEngine())
    with TestClient(broken) as client:
        failed = client.post("/transcribe", files={"file": ("lecture.wav", sine_wav(), "audio/wav")})
    assert failed.status_code == 500
    assert failed.json()["detail"] == "Transcription failed."
    assert "weights" not in failed.text


def test_websocket_streams_a_final_segment():
    engine = FakeEngine()
    app = create_app(settings=_settings(), engine=engine)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe") as socket:
            socket.send_json({"type": "start", "sample_rate": 16000})
            ready = socket.receive_json()
            from support import sine_pcm16

            socket.send_bytes(sine_pcm16(1.0))
            socket.send_json({"type": "stop"})
            messages = _messages(socket)
    assert ready["type"] == "ready"
    assert ready["encoding"] == "pcm_s16le"
    assert ready["glossary_terms"] == 3
    kinds = [message["type"] for message in messages]
    assert "segment" in kinds
    assert kinds[-1] == "done"
    segment = next(message for message in messages if message["type"] == "segment")
    assert segment["final"] is True
    assert segment["id"].endswith("-seg-1")
    assert segment["id"].startswith("s")
    assert segment["text"] == engine.text
    assert engine.calls[0]["vad"] is False


def test_websocket_rejects_audio_before_start_and_odd_pcm():
    app = create_app(settings=_settings(), engine=FakeEngine())
    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe") as socket:
            socket.send_bytes(b"\x00\x00")
            early = socket.receive_json()
            socket.send_json({"type": "start"})
            socket.receive_json()
            socket.send_bytes(b"\x00\x01\x02")
            odd = socket.receive_json()
            socket.send_json({"not": "an event"})
            unknown = socket.receive_json()
            socket.send_json({"type": "stop"})
            done = socket.receive_json()
    assert early["type"] == "error"
    assert odd["type"] == "error"
    assert unknown["type"] == "error"
    assert done["type"] == "done"


def test_missing_glossary_still_starts(tmp_path, caplog):
    missing = tmp_path / "deep-learning.csv"
    with caplog.at_level(logging.WARNING, logger="asr.glossary"):
        app = create_app(
            settings=Settings.from_env({"ASR_GLOSSARY_PATH": str(missing)}),
            engine=FakeEngine(),
        )
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["glossary_terms"] == 0
    assert "ASR_GLOSSARY_PATH" in caplog.text


def test_bad_glossary_refuses_to_start(tmp_path):
    bad = tmp_path / "course.csv"
    bad.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(GlossaryFormatError):
        create_app(settings=Settings.from_env({"ASR_GLOSSARY_PATH": str(bad)}), engine=FakeEngine())


def test_two_live_sessions_do_not_reuse_segment_ids():
    app = create_app(settings=_settings(), engine=FakeEngine())
    ids = []
    with TestClient(app) as client:
        for _ in range(2):
            with client.websocket_connect("/ws/transcribe") as socket:
                socket.send_json({"type": "start", "sample_rate": 16000})
                socket.receive_json()
                from support import sine_pcm16

                socket.send_bytes(sine_pcm16(1.0))
                socket.send_json({"type": "stop"})
                messages = _messages(socket)
            ids.append(next(message["id"] for message in messages if message["type"] == "segment"))
    assert ids[0] != ids[1]
    assert ids[0].endswith("-seg-1") and ids[1].endswith("-seg-1")


def test_batch_windows_keep_offsets_and_release_the_model_between_them():
    from asr.audio import wav_bytes_from_pcm16
    from support import sine_pcm16

    engine = FakeEngine()
    app = create_app(settings=_settings(batch_window_s=1, max_audio_seconds=30), engine=engine)
    wav = wav_bytes_from_pcm16(sine_pcm16(2.0))
    with TestClient(app) as client:
        created = client.post(
            "/uploads",
            content=wav,
            headers={"content-type": "application/octet-stream", "x-filename": "two.wav"},
        )
        assert created.status_code == 200
        upload_id = created.json()["id"]
        first = client.post(f"/uploads/{upload_id}/next")
        second = client.post(f"/uploads/{upload_id}/next")
        done = client.post(f"/uploads/{upload_id}/next")
        client.delete(f"/uploads/{upload_id}")
    assert first.json()["segments"][0]["start"] < 1
    assert second.json()["segments"][0]["start"] >= 1
    assert done.json()["done"] is True
    assert len(engine.calls) == 2


def test_overlong_recording_is_rejected_and_does_not_stick_the_lock(tmp_path):
    from asr.audio import wav_bytes_from_pcm16
    from support import sine_pcm16

    engine = FakeEngine()
    app = create_app(settings=_settings(max_audio_seconds=1), engine=engine)
    long_wav = wav_bytes_from_pcm16(sine_pcm16(2.0))
    short_wav = wav_bytes_from_pcm16(sine_pcm16(0.4))
    with TestClient(app) as client:
        rejected = client.post("/transcribe", files={"file": ("long.wav", long_wav, "audio/wav")})
        assert rejected.status_code == 413
        detail = rejected.json()["detail"]
        assert detail["code"] == "audio_too_long"
        assert detail["message"] == "录音太长，请分段上传"
        assert detail["max_seconds"] == 1
        accepted = client.post("/transcribe", files={"file": ("short.wav", short_wav, "audio/wav")})
    assert accepted.status_code == 200
    assert engine.calls
    assert accepted.json()["segments"][0]["id"].endswith("-seg-1")


def test_settings_defaults_and_validation():
    settings = Settings.from_env({})
    assert settings.model == "base.en"
    assert settings.compute_type == "int8"
    assert settings.cpu_threads == 2
    assert settings.glossary_path.as_posix() == "glossary/deep-learning.csv"
    assert settings.max_audio_seconds == 7200
    assert settings.cors_origins == ("*",)
    custom = Settings.from_env(
        {
            "ASR_MODEL": "tiny.en",
            "ASR_CPU_THREADS": "2",
            "ASR_GLOSSARY_PATH": "glossary/nlp.csv",
            "ASR_CORS_ORIGINS": "http://a, http://b",
            "ASR_LOCAL_FILES_ONLY": "1",
        }
    )
    assert custom.model == "tiny.en"
    assert custom.glossary_path.as_posix() == "glossary/nlp.csv"
    assert custom.cors_origins == ("http://a", "http://b")
    assert custom.local_files_only is True
    with pytest.raises(ValueError):
        Settings.from_env({"ASR_CPU_THREADS": "nope"})
    with pytest.raises(ValueError):
        Settings.from_env({"ASR_CPU_THREADS": "0"})


def test_faster_whisper_import_is_inside_the_loader():
    from pathlib import Path

    import asr.engine as engine

    source = Path(engine.__file__).read_text(encoding="utf-8")
    before_loader = source.split("def ensure_loaded", 1)[0]
    assert "faster_whisper" not in before_loader
    assert "from faster_whisper import WhisperModel" in source
