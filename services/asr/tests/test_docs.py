"""Guards for the contract, the private ports, and the weight/audio ignore rules."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_translate_flow_points_at_annas_readme():
    text = (ROOT / "docs" / "translate-contract.md").read_text(encoding="utf-8")
    assert "services/translate/README.md" in text
    assert "POST /translate" in text or "`POST`" in text
    assert "http://translate:8080" in text
    assert "503" in text
    assert "glossary" in text
    assert "segments" in text
    assert "2000" in text
    assert "seg-1" in text


def test_glossary_path_and_headers_are_documented():
    glossary = (ROOT / "docs" / "glossary.md").read_text(encoding="utf-8")
    asr = (ROOT / "docs" / "asr.md").read_text(encoding="utf-8")
    for text in (glossary, asr):
        assert "ASR_GLOSSARY_PATH" in text
        assert "glossary/deep-learning.csv" in text
        assert "en,zh,note" in text
        assert "英文,中文,备注" in text


def test_deploy_keeps_asr_and_translate_off_the_public_internet():
    docs = (ROOT / "docs" / "deploy.md").read_text(encoding="utf-8")
    compose = (ROOT / "deploy" / "docker-compose.yml").read_text(encoding="utf-8")
    assert "security group" in docs
    assert "127.0.0.1" in docs
    assert "127.0.0.1:8081" in compose
    assert "127.0.0.1:8001" in compose
    assert "0.0.0.0:8080" in compose
    assert '"8081:8080"' not in compose
    assert compose.count("../glossary:/glossary:ro") >= 2
    assert "TRANSLATE_GLOSSARY_DIR: /glossary" in compose
    assert "http://translate:8080" in compose
    assert "TRANSLATE_INTRA_THREADS:-1" in compose.replace("}", "")
    assert "TRANSLATE_CPUS:-1" in compose.replace("}", "")
    assert "TRANSLATE_MEM_LIMIT:-640m" in compose.replace("}", "")
    assert "WEB_ACCESS_TOKEN:?" in compose.replace("}", "") or "WEB_ACCESS_TOKEN:" in compose
    assert "one 2-vCPU" in docs or "One Alibaba" in docs
    assert "swapfile" in docs
    assert "mainland" in docs.lower() or "Mainland" in docs


def test_ci_requirements_do_not_install_whisper():
    text = (ROOT / "services" / "asr" / "requirements-ci.txt").read_text(encoding="utf-8").lower()
    assert "faster-whisper" not in text
    assert "ctranslate2" not in text


def test_gitignore_blocks_weights_and_recordings():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for token in ("*.wav", "*.mp3", "*.m4a", "*.onnx", "*.pt", "recordings/", "uploads/", "deploy/data/"):
        assert token in text


def test_docs_index_lists_every_doc():
    docs = ROOT / "docs"
    index = (docs / "README.md").read_text(encoding="utf-8")
    names = sorted(path.name for path in docs.glob("*.md") if path.name != "README.md")
    assert names
    for name in names:
        assert f"]({name})" in index
    assert "results.md" in index
    assert "deploy.md" in index
    assert "translate-contract.md" in index


def test_deploy_documents_results_retention_and_token_logs():
    docs = (ROOT / "docs" / "deploy.md").read_text(encoding="utf-8")
    results = (ROOT / "docs" / "results.md").read_text(encoding="utf-8")
    compose = (ROOT / "deploy" / "docker-compose.yml").read_text(encoding="utf-8")
    example = (ROOT / "deploy" / ".env.example").read_text(encoding="utf-8")
    assert "docker compose" in docs and "logs web" in docs
    assert "RESULT_RETENTION_DAYS" in docs
    assert "RESULT_DIR" in docs
    assert "/data/results" in docs
    assert "RESULT_RETENTION_DAYS" in results
    assert "web-results:/data/results" in compose
    assert "${RESULT_DIR:-/data/results}" in compose
    assert "${RESULT_RETENTION_DAYS:-7}" in compose
    assert "${RESULT_RETENTION_DAYS:-7}" in example
    assert "${UPLOAD_MAX_BODY:-}" in example
    ignore = (ROOT / "apps" / "web" / ".dockerignore").read_text(encoding="utf-8")
    for token in (".env", "node_modules", "*.upload"):
        assert token in ignore
    for dockerfile in (ROOT / "apps" / "web" / "Dockerfile", ROOT / "services" / "asr" / "Dockerfile"):
        text = dockerfile.read_text(encoding="utf-8")
        assert "\nUSER " in text
    req = (ROOT / "services" / "asr" / "requirements.txt").read_text(encoding="utf-8")
    ci = (ROOT / "services" / "asr" / "requirements-ci.txt").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    assert "uvicorn[standard]==0.34.0" in req
    assert "uvicorn[standard]==0.34.0" in ci
    assert "av==" in ci
    assert "ffmpeg" in workflow
    server = (ROOT / "apps" / "web" / "server.py").read_text(encoding="utf-8")
    assert "TRANSCRIBE_MAX_BODY" not in server
    assert "UPLOAD_MAX_BODY" in server
