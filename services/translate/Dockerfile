# CPU-only English-to-Chinese translation service.
# Build from the repository root:
#   docker build -f services/translate/Dockerfile -t translator-edu-translate:cpu .
#
# Model weights are not copied into the image. Mount the directory from
# scripts/download_model.py at /models/opus-mt-en-zh-ct2.

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSLATE_BACKEND=ctranslate2 \
    TRANSLATE_MODEL_DIR=/models/opus-mt-en-zh-ct2 \
    TRANSLATE_GLOSSARY_DIR=/glossary \
    TRANSLATE_INTRA_THREADS=2 \
    TRANSLATE_INTER_THREADS=1 \
    TRANSLATE_BEAM_SIZE=1 \
    TRANSLATE_COMPUTE_TYPE=int8

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin translate

COPY services/translate/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY services/translate/app /app/app
COPY glossary /glossary

USER translate
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
