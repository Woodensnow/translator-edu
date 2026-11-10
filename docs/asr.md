# ASR service

CPU transcription for English lectures. The model is faster-whisper `base.en`, int8, one worker, `cpu_threads` 2, `beam_size` 1. English-only weights are a better fit than the multilingual checkpoint for this class, and base is the smallest size that still holds technical words when the glossary is in the prompt. `tiny.en` is the fallback if a 2-vCPU host cannot stay ahead of the lecture (`ASR_MODEL=tiny.en`). Measured speed is in [deploy.md](deploy.md).

Weights are not in git. On the host, before the first lecture:

```bash
cd services/asr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ASR_MODEL=base.en ASR_MODEL_DIR=/var/lib/translator-edu/models \
  python scripts/download_model.py
```

The server also downloads on the first transcription when `ASR_LOCAL_FILES_ONLY` is unset. `scripts/bench_rtf.py` repeats a short clip on two pinned CPUs and prints the real-time factor.

## Run

```bash
cd services/asr
PYTHONPATH=src ASR_GLOSSARY_PATH=../../glossary/deep-learning.csv \
  ASR_CPU_THREADS=2 python -m asr
```

The process listens on `ASR_PORT` (8001). Compose publishes that only on `127.0.0.1` for a debug curl on the ECS host. The lecturer's browser reaches it through the captions page.

| Variable | Default | Role |
| --- | --- | --- |
| `ASR_MODEL` | `base.en` | faster-whisper name. Use `tiny.en` if memory is tight. Live mic, when it becomes the default later, should use `tiny.en`. |
| `ASR_COMPUTE_TYPE` | `int8` | CTranslate2 compute type |
| `ASR_CPU_THREADS` | `2` | Must match the vCPU count. Unset, the library uses 4 threads. |
| `ASR_MODEL_DIR` | Hugging Face cache | Weight directory |
| `ASR_GLOSSARY_PATH` | `glossary/deep-learning.csv` | Course CSV with header `en,zh,note` or `英文,中文,备注`. See [glossary.md](glossary.md). |
| `ASR_WINDOW_SECONDS` | `4` | Live window. Latency is about this plus the model time. |
| `ASR_PARTIAL_SECONDS` | `0` | Set to `2` only when the real-time factor has spare room. |
| `ASR_MAX_UPLOAD_MB` | `100` | Byte cap while the body is streamed to disk. |
| `ASR_MAX_AUDIO_SECONDS` | `7200` | Reject longer recordings. The page says 录音太长，请分段上传. |
| `ASR_BATCH_WINDOW_SECONDS` | `45` | Decode and transcribe this much at a time. |
| `ASR_CPU_THREADS` | `2` | Threads inside one window. Translate uses 1 and runs between windows. |
| `ASR_CORS_ORIGINS` | `*` | Unused by the browser when the page proxies. Leave the port unpublished. |

## Browser protocol

`WS /ws/transcribe`

The client sends `{"type":"start","sample_rate":16000}`, then binary PCM16 little-endian mono, then `{"type":"stop"}`. The server answers `ready`, then final segments:

```json
{"type":"segment","id":"seg-1","start":12.4,"end":15.1,"text":"Backpropagation uses gradient descent.","final":true}
```

`{"type":"status","lagging":true,"rtf":1.2}` means that window was slower than real time. The page stops streaming and transcribes the recording at the end.

`POST /transcribe` with a multipart field `file` returns:

```json
{"language":"en","text":"...","segments":[{"id":"seg-1","start":0.1,"end":3.8,"text":"..."}]}
```

WAV is decoded in-process. Other containers need PyAV, which ships with faster-whisper. Batch transcription may use the bundled Silero VAD (`ASR_BATCH_VAD=0` turns it off). That file is part of the Python package, not a downloaded weight.

`GET /health` does not load the model.
