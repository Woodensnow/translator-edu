import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_git_ignores_weights_and_audio_but_not_the_service():
    ignored = [
        "services/translate/models/opus-mt-en-zh-ct2/model.bin",
        "services/translate/models/opus-mt-en-zh-ct2/source.spm",
        "services/translate/models/opus-mt-en-zh-ct2/runtime.json",
        "lecture.mp3",
        "lecture.m4a",
        "recordings/lecture.wav",
        "weights.safetensors",
        "checkpoint.pt",
    ]
    tracked = [
        "glossary/deep-learning.csv",
        "services/translate/app/main.py",
        "services/translate/README.md",
        "services/translate/scripts/download_model.py",
    ]
    for rel in ignored:
        result = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT, check=False)
        assert result.returncode == 0, rel
    for rel in tracked:
        result = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT, check=False)
        assert result.returncode == 1, rel


def test_ci_does_not_download_weights():
    workflow = (ROOT / ".github" / "workflows" / "translate.yml").read_text(encoding="utf-8")
    assert "requirements-ci.txt" in workflow
    assert "download_model" not in workflow
    assert "benchmark_sentinels" not in workflow
    assert "TRANSLATE_BACKEND: mock" in workflow
    ci = (ROOT / "services" / "translate" / "requirements-ci.txt").read_text(encoding="utf-8").lower()
    for package in ("ctranslate2", "torch", "transformers", "sentencepiece", "huggingface"):
        assert package not in ci
    dockerfile = (ROOT / "services" / "translate" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY services/translate/models" not in dockerfile
    assert "TRANSLATE_MODEL_DIR" in dockerfile


def test_compose_binds_localhost_and_mounts_glossary():
    text = (ROOT / "services" / "translate" / "docker-compose.yml").read_text(encoding="utf-8")
    assert "127.0.0.1:8081:8080" in text
    assert "glossary:/glossary:ro" in text
    assert '"8081:8080"' not in text


def test_download_script_help_does_not_fetch_weights():
    script = ROOT / "services" / "translate" / "scripts" / "download_model.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "opus-mt-en-zh" in result.stdout
    assert "int8" in result.stdout
    assert "float16" in result.stdout


def test_committed_sentinel_benchmark_matches_the_code():
    from app.glossary import SENTINEL_PREFIX

    data = json.loads(
        (ROOT / "services" / "translate" / "benchmarks" / "sentinel-results.json").read_text(
            encoding="utf-8"
        )
    )
    by_name = {item["name"]: item for item in data["schemes"]}
    assert data["chosen_prefix"] == SENTINEL_PREFIX == "ZXQ"
    assert data["chosen_name"] == "zxq"
    assert by_name["zxq"]["exact"] == max(item["exact"] for item in data["schemes"])
    assert by_name["brackets"]["exact"] == 0
    assert by_name["zxq"]["restored"] == 1
    assert len(data["per_sentence"]) == 12
    assert all(row["baseline_ms"] > 0 and row["chosen_ms"] > 0 for row in data["per_sentence"])
