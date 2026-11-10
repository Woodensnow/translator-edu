FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY scripts ./scripts

# uid 10001 owns /models. Glossary mounts stay read-only and world-readable.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin asr \
    && mkdir -p /models/hf \
    && chown -R asr:asr /models

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    ASR_MODEL=base.en \
    ASR_COMPUTE_TYPE=int8 \
    ASR_CPU_THREADS=2 \
    ASR_MODEL_DIR=/models \
    ASR_GLOSSARY_PATH=/glossary/deep-learning.csv \
    ASR_MAX_AUDIO_SECONDS=7200 \
    OMP_NUM_THREADS=2 \
    ASR_HOST=0.0.0.0 \
    ASR_PORT=8001 \
    HF_HOME=/models/hf \
    HUGGINGFACE_HUB_CACHE=/models/hf

USER asr
EXPOSE 8001
VOLUME ["/models"]

CMD ["python", "-m", "asr"]
