FROM python:3.11-slim

WORKDIR /app
COPY index.html server.py ./
COPY css ./css
COPY js ./js

ENV WEB_HOST=0.0.0.0 \
    WEB_PORT=8080 \
    TRANSLATE_URL=http://translate:8080 \
    ASR_HTTP_URL=http://asr:8001 \
    ASR_WS_URL=ws://asr:8001/ws/transcribe \
    COURSE_ID=deep-learning \
    PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["python", "server.py"]
