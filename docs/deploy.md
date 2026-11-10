# Deploy

One Alibaba Cloud ECS instance, 2 vCPU, about 2 GB RAM, no GPU, in mainland China. Web, ASR, and translate run on that host together.

```bash
openssl rand -hex 32
cp deploy/.env.example deploy/.env
# paste the token into deploy/.env as WEB_ACCESS_TOKEN
docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d --build
```

Compose reads `deploy/.env` because that file is passed with `--env-file` and because the project directory is the directory of the first `-f` file (`deploy/`), not the repo root. Do not commit `deploy/.env`.

Unset and empty thread variables resolve to the single-host defaults (`ASR_CPU_THREADS` 2, `TRANSLATE_INTRA_THREADS` 1, `TRANSLATE_INTER_THREADS` 1). Unset and empty `WEB_ACCESS_TOKEN` both fail `docker compose config` instead of booting with a blank password.

`docker compose config` still passes if the token is the `.env.example` placeholder or shorter than 32 characters. The web process exits, and `restart: unless-stopped` starts it again. The same log line is where a result-directory permission error shows up. Read it with:

```bash
docker compose -f deploy/docker-compose.yml logs web
```

## What is reachable

Only the captions page is on the public interface (`WEB_PUBLISH`, default `0.0.0.0:8080`). ASR is `127.0.0.1:8001` and translate is `127.0.0.1:8081`, for a curl from the ECS host. The browser never calls them. The page proxies, and it asks for `WEB_ACCESS_TOKEN` once. The cookie is HttpOnly and SameSite=Strict. It is not marked Secure, because v1 is plain HTTP. Add Secure when HTTPS lands.

The security group should allow the web port only. Do not authorize 8001 or 8081 from the internet.

v1 is reached by the bare IP and port over HTTP. A domain on 443 in mainland China needs ICP filing. A Hong Kong region would not. The region detail beyond "mainland, no filing yet" can still be confirmed with Tim. Caddy and HTTPS are a later PR. Origin checks beyond the token wait for that PR too. Wrong-token lockout (5 failures, then exponential backoff, 429) is already on the page.

The microphone needs a secure context, so live captions are not the default path. Upload a recording. Live mic stays in the page for later; when it becomes the default, run it with `ASR_MODEL=tiny.en`.

## One job at a time

An upload is saved to disk, then transcribed in about 45-second windows. Each window is translated before the next window starts, so the two models take turns on the two cores. The page polls and draws bilingual rows as windows finish (`已处理 12/90 分钟`). When the job is done it offers a bilingual `.txt` and `.srt`. A second upload waits in the queue.

`ASR_CPU_THREADS` defaults to 2. Translate defaults to `cpus: 1` (`TRANSLATE_CPUS`) and `mem_limit: 640m` (`TRANSLATE_MEM_LIMIT`). An empty or unset value uses those defaults. The two models take turns on a chunk, so those pools are not stacked. See the measured section for why 2 remains the ASR default. Anna's real-model run peaked around 227 MiB, so 640 MB leaves headroom for the runtime.

A recording longer than `ASR_MAX_AUDIO_SECONDS` (default 7200, two hours) is rejected. The page says `录音太长，请分段上传` and shows the limit. It is not cut down and returned as a short success.

## Glossary mount

Both services mount `glossary/` read-only at `/glossary`:

- ASR: `ASR_GLOSSARY_PATH=/glossary/deep-learning.csv`
- translate: `TRANSLATE_GLOSSARY_DIR=/glossary`

The translate image also copies that directory at build time. The mount hides the copy, so an edit on the host is what the process reads. The glossary mount stays read-only. ASR runs as uid 10001 and web as uid 10002, and both can read a normal `644` file.

## Saved results

A finished upload is written to `RESULT_DIR` (default `/data/results`) before the page is told the job is done. The file is the bilingual segments. The page lists name, date, and duration, reopens a result, and downloads `captions.txt` and `captions.srt`. Delete removes that file. Audio temp files are still deleted when processing finishes; they are not in this directory. Only the latest few jobs stay in the web process. The files survive a container restart. Details are in [results.md](results.md).

Compose mounts the named volume `web-results` at `/data/results`. To keep the files on the host instead, replace that volume with a bind mount of `deploy/data` (gitignored):

```yaml
- ./data/results:/data/results
```

```bash
mkdir -p deploy/data/results
sudo chown -R 10002:10002 deploy/data
```

`RESULT_RETENTION_DAYS` defaults to 7. Expired files are removed when web starts and about once an hour. `0` keeps every result until someone deletes it. An empty or unset value uses 7.

If an older `asr-models` volume was created while ASR ran as root, the non-root process cannot write weights. Fix it once:

```bash
docker compose -f deploy/docker-compose.yml run --user root --rm asr chown -R asr:asr /models
```

Whisper weights go in the `asr-models` volume or via `services/asr/scripts/download_model.py`. Opus-mt weights are `services/translate/models/opus-mt-en-zh-ct2`, built with the translate README. Both are gitignored. Until the opus-mt directory exists, translate's `/health` stays 503 and Chinese columns fall back to English.

## Memory

Plan for about 2 GB of RAM.

| Container | mem_limit | Why |
| --- | --- | --- |
| asr | 768 MB | `base.en` int8 resident is about 400 MB. A window of audio is a few MB, not the whole lecture. |
| translate | 640 MB | One CPU, matching one translate thread. Anna measured the real int8 model at about 185 MiB idle and 227 MiB peak, with one thread averaging 93 ms per sentence (p95 291 ms). 640 MB stays above that peak. |
| web | 128 MB | Proxy, queue, and static files. |

The sum is 1536 MB, under 1.6 GB, so the OS still has a little room. An overrun hits that container's limit instead of the whole machine. If the box is tight, set `ASR_MODEL=tiny.en` (about 300 MB). A two-hour file decoded as float32 16 kHz mono would be about 460 MB if it were held at once. Uploads are streamed to a temp file and decoded one window at a time. The temp file is deleted when the job ends.

Add a 2 GB swapfile on the host:

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Measured speed

Real-time factor is model time divided by audio length. Lower than 1 means the window finishes before the next window of speech would. Numbers below are filled in after a run pinned with `taskset -c 0,1` on this VM (Intel Xeon, 2400 MHz).

Both processes were pinned with `taskset -c 0,1` on this VM (Intel Xeon, 2400 MHz). The recording is the public 11-second English clip repeated to 10 minutes (14 windows of 45 seconds). After each window the client called the translate service, which is how the upload queue alternates the two models. Translate ran as the mock backend, so the RSS column is the process without opus-mt weights. Anna later measured the real int8 model at about 185 MiB idle and 227 MiB peak, which is what the 640 MB limit is sized for.

| ASR threads | ASR time | ASR RTF | ASR peak RSS | Translate peak RSS | Combined peak RSS |
| --- | --- | --- | --- | --- | --- |
| 1 | 32.1 s | 0.054 | 477 MB | 43 MB mock | 520 MB |
| 2 | 21.4 s | 0.036 | 476 MB | 43 MB mock | 519 MB |

`ASR_CPU_THREADS=2` is the default. It is about 1.5× faster than one thread, and the RSS stays the same. Translate's one intra thread runs after the window, so the second ASR thread and the translate thread take turns. With Anna's 227 MiB translate peak, plan on about 700 MB resident before the OS and a decoded window. That fits a 2 GB instance with the swapfile above. RTF 0.036 keeps up with a recording. Use `tiny.en` when the 768 MB ASR limit is too tight for the box.

An earlier solo run of the 11-second and 33-second clips, also pinned to two cores, was RTF 0.042 and 0.029 for `base.en` and 0.031 and 0.016 for `tiny.en`.

Later: a second host is not the v1 layout. If the lecture outgrows this machine, split translate onto another box and point `TRANSLATE_URL` at it. Do not publish that port on `0.0.0.0`.
