# Deploy

Two Alibaba Cloud ECS instances, 2 vCPU, no GPU. Whisper and the translation model should not share those two cores during a lecture. The Compose file can still start every service, which is useful on a bigger box or for a dry run.

| Host | Services | Cores during a lecture |
| --- | --- | --- |
| Transcription | `asr`, `web` | ASR uses both. The page is idle. |
| Translation | `translate` | opus-mt int8, `intra_threads` 2 |

```bash
# transcription host
docker compose -f deploy/docker-compose.yml up -d --build asr web

# translation host, private address only
TRANSLATE_PUBLISH=10.0.0.12:8080 \
  docker compose -f deploy/docker-compose.yml up -d --build translate
```

On the transcription host set `TRANSLATE_URL=http://10.0.0.12:8080`. When all three containers share one Compose network, leave the default `http://translate:8080`.

Copy `deploy/.env.example` to `deploy/.env` and edit the private address. Do not commit `.env`.

## What is reachable

Only the captions page is bound for the classroom network (`WEB_PUBLISH`, default `0.0.0.0:8080`).

ASR is published as `127.0.0.1:8001` and translate as `127.0.0.1:8081`. Those mappings exist so someone logged into the ECS host can curl a health check. They are not on the public interface. A split translation host may set `TRANSLATE_PUBLISH` to that instance's **private** IP and port `8080`. Do not set it to `0.0.0.0`.

The ECS security group should allow inbound traffic on the web port only (8080, or 443 if a TLS proxy is added later). Do not authorize 8001 or 8081 from `0.0.0.0/0`. If translate runs on the other instance, authorize its private port from the transcription host's private IP only. ASR does not need a security-group rule: the page proxies to it over the Compose network.

Translate does not authenticate callers and answers `Access-Control-Allow-Origin: *`. Publishing it would let anyone on the internet use the CPU. The browser never calls it. The page forwards `POST /translate` on the server side.

Microphone capture needs a secure context. `http://127.0.0.1` works for a local trial. A classroom laptop opening `http://<public-ip>:8080` will not get a microphone until the web port is behind TLS. Upload still works over plain HTTP. The security group note still applies: open that web port, not the others.

## Glossary mount

Both services mount the repo `glossary/` directory read-only at `/glossary`:

- ASR: `ASR_GLOSSARY_PATH=/glossary/deep-learning.csv`
- translate: `TRANSLATE_GLOSSARY_DIR=/glossary`

Editing `glossary/deep-learning.csv` on the host is what the containers read. The translate image also copies the directory at build time; the mount hides that copy so a later edit is visible without a rebuild. Translate reloads a changed file itself. ASR reloads when the modification time changes.

The converted opus-mt directory is mounted read-only from `services/translate/models/opus-mt-en-zh-ct2`. Build it with the translate README's download script. It is gitignored. Until those files exist, translate's `/health` stays 503 and the page shows English in both columns.

Whisper weights download into the `asr-models` volume (`ASR_MODEL_DIR=/models`) on first use, or via `services/asr/scripts/download_model.py`.

## Resources

Rough working set on a 2-vCPU instance:

| Process | RAM | Notes |
| --- | --- | --- |
| ASR `base.en` int8 | about 400 MB measured, plan for 1 GB | `cpu_threads=2`, beam 1. Prefer a 4 GB instance so an upload and the OS fit. |
| ASR `tiny.en` int8 | about 300 MB measured | Switch with `ASR_MODEL=tiny.en` if live captions report lag. |
| translate int8 | about 0.5 GB | Anna's figure for opus-mt. One uvicorn worker. |
| web | under 50 MB | Proxy and static files. |

`cpus: "2"` on ASR and translate matches the instance. On a single 2-vCPU host those limits contend; run the split above for a real lecture.

## Measured speed

Real-time factor is model time divided by audio length, pinned to two cores (`taskset -c 0,1`, `ASR_CPU_THREADS=2`). Lower than 1 means the window finishes before the next window of speech arrives. The clip is the public short English sample used by the upstream Whisper tests, once and then repeated three times. This VM is an Intel Xeon (family 6, model 207, about 2.4 GHz class), not the ECS instance. Treat the number as an upper bound on speed for a similar core; an older ECS generation should try `tiny.en` if live captions report `lagging`.

Pinned with `taskset -c 0,1` on this VM (Intel Xeon, 2400 MHz, two cores visible). `scripts/bench_rtf.py` warmed the model, then timed a public 11-second English clip and the same clip repeated to 33 seconds. Both sizes produced the sentence. `base.en` is the default: the real-time factor is far below 1, so the larger English model is affordable and technical terms have more to gain from it than from `tiny.en`.

| Model | Audio | Elapsed | Real-time factor | Peak RSS |
| --- | --- | --- | --- | --- |
| `tiny.en` int8 | 11 s | 0.34 s | 0.031 | 276 MB |
| `tiny.en` int8 | 33 s | 0.52 s | 0.016 | 293 MB |
| `base.en` int8 | 11 s | 0.46 s | 0.042 | 368 MB |
| `base.en` int8 | 33 s | 0.96 s | 0.029 | 389 MB |

A 4-second live window at a real-time factor of 0.04 costs about 0.2 seconds of compute, so the wait is the window, not the model. An older ECS core can be several times slower than this Xeon and still finish inside the window. If a host reports `lagging`, set `ASR_MODEL=tiny.en`. Leave `ASR_PARTIAL_SECONDS` at 0 unless you have measured spare room; `2` is reasonable on a core this fast.

Reproduce:

```bash
ASR_MODEL=base.en ASR_MODEL_DIR=/var/lib/translator-edu/models ASR_CPU_THREADS=2 \
  taskset -c 0,1 python services/asr/scripts/bench_rtf.py --audio lecture.wav --threads 2
```

Do not commit the wav. The script refuses nothing by itself; `.gitignore` ignores audio and weight files.
