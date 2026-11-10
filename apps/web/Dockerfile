FROM python:3.11-slim

# uid 10002 owns /data/results. A fresh named volume inherits that owner.
RUN useradd --create-home --uid 10002 --shell /usr/sbin/nologin web \
    && mkdir -p /data/results \
    && chown -R web:web /data/results

WORKDIR /app
COPY . .

ENV WEB_HOST=0.0.0.0 \
    WEB_PORT=8080 \
    TRANSLATE_URL=http://translate:8080 \
    ASR_HTTP_URL=http://asr:8001 \
    ASR_WS_URL=ws://asr:8001/ws/transcribe \
    COURSE_ID=deep-learning \
    RESULT_DIR=/data/results \
    RESULT_RETENTION_DAYS=7 \
    PYTHONUNBUFFERED=1

USER web
EXPOSE 8080
CMD ["python", "server.py"]
