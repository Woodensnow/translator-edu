# Translation service

English-to-Chinese captions for a live lecture. `POST /translate` takes English text or ASR segments with timestamps and returns Chinese text plus bilingual segments. A course glossary is applied around the model so domain terms stay fixed.

No shared HTTP contract was in `docs/` when this service was added. This file is the request and response contract until that document exists. Unknown JSON fields are ignored.

The service does not authenticate callers. Keep it on the lecture network.

## API

`GET /health` returns 200 when the glossary and backend are ready, and 503 otherwise.

```json
{
  "status": "ok",
  "ready": true,
  "backend": "ctranslate2",
  "model": "Helsinki-NLP/opus-mt-en-zh",
  "version": "0.1.0",
  "glossaries": ["deep-learning"],
  "default_glossary": "deep-learning",
  "detail": null
}
```

`GET /glossary` lists course files. `GET /glossary/deep-learning` returns the same `en`, `zh`, `note` rows as `glossary/deep-learning.csv`.

`POST /translate`

Timestamps are seconds. When `segments` is non-empty it is the translation unit, so ASR timings stay aligned, and `text` is ignored. Without segments, `text` becomes one segment. `source_lang` must be English and `target_lang` Chinese (`en` and `zh` by default). The joined `translation` is the segment translations separated by newlines.

```json
{
  "glossary": "deep-learning",
  "segments": [
    {
      "id": "seg-1",
      "start": 12.4,
      "end": 15.1,
      "text": "Backpropagation uses gradient descent."
    }
  ]
}
```

```json
{
  "source_lang": "en",
  "target_lang": "zh",
  "glossary": "deep-learning",
  "model": "Helsinki-NLP/opus-mt-en-zh",
  "backend": "ctranslate2",
  "translation": "反向传播使用梯度下降。",
  "segments": [
    {
      "id": "seg-1",
      "start": 12.4,
      "end": 15.1,
      "source": "Backpropagation uses gradient descent.",
      "translation": "反向传播使用梯度下降。",
      "terms": [
        {
          "en": "backpropagation",
          "zh": "反向传播",
          "note": "",
          "matched": "Backpropagation"
        },
        {
          "en": "gradient descent",
          "zh": "梯度下降",
          "note": "",
          "matched": "gradient descent"
        }
      ]
    }
  ]
}
```

The Chinese sentence above is the shape of a model response, not a captured fixture. `backend` is `mock` or `ctranslate2`. In mock mode the glossary Chinese is still substituted and the other words stay in English, which is what the tests lock.

| status | when |
| --- | --- |
| 422 | missing text and segments, bad language, bad timestamps, or over 20,000 characters / 200 segments |
| 404 | unknown `glossary` id. `detail.glossaries` lists the files on disk |
| 503 | weights are not converted yet, or the glossary directory cannot be read |

```bash
curl -s http://127.0.0.1:8080/translate \
  -H 'content-type: application/json' \
  -d '{"text":"Backpropagation uses gradient descent.","glossary":"deep-learning"}'
```

## Glossary

Terms come from `glossary/*.csv` (`en,zh,note`). See `glossary/README.md` for how to add a row or a course. The default id is `deep-learning`.

Hits are replaced with sentinels, the sentence is translated, and each sentinel is swapped for its Chinese term. If a sentinel does not survive the model, the sentence is translated in pieces around the locked terms so the Chinese is still inserted. The model is not the only thing enforcing terminology.

## Model

[Helsinki-NLP/opus-mt-en-zh](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) (CC-BY-4.0), a Marian English-to-Chinese model of about 111M parameters (6 encoder layers, 6 decoder layers, hidden size 512). The fp32 checkpoint is about 300MB. This service runs the **CTranslate2 int8** conversion on CPU.

That pair fits one 2-core machine: published CTranslate2 int8 numbers for Opus-MT are on the order of 0.5GB of RAM, and a caption sentence is a small batch. Beam size defaults to 1 for live captions. `intra_threads` is 2 and `inter_threads` is 1, and the process uses a single uvicorn worker so the model is loaded once.

Larger checkpoints (NLLB-600M, `opus-mt-tc-big-en-zho`, mBART) are a poor fit for this CPU budget. float16 is not the default because these servers have no GPU. The runtime never selects CUDA.

This dedicated en-zh checkpoint does not take a target-language token. Do not prefix text with `>>zho<<`.

## Download and convert weights

Do this once, on a machine with at least 4GB of RAM. Do not commit the output. Conversion is not part of CI or the request path.

```bash
cd services/translate
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-convert.txt
python scripts/download_model.py
```

The script writes `services/translate/models/opus-mt-en-zh-ct2/` (gitignored) and prints the directory size. Copy that directory to the server. Equivalent converter invocation:

```bash
ct2-transformers-converter \
  --model Helsinki-NLP/opus-mt-en-zh \
  --output_dir models/opus-mt-en-zh-ct2 \
  --quantization int8 \
  --copy_files source.spm target.spm vocab.json tokenizer_config.json \
  --low_cpu_mem_usage
```

If you use the CLI, also write `runtime.json` in that directory with `"compute_type": "int8"`, or set `TRANSLATE_COMPUTE_TYPE=int8`. `--quantization float32` is available when int8 is not; float16 is intentionally not offered.

## Run

Mock backend, no weights, glossary still applied:

```bash
cd services/translate
pip install -r requirements-ci.txt
TRANSLATE_BACKEND=mock uvicorn app.main:app --host 0.0.0.0 --port 8080
```

CPU model, after the conversion above. Install runtime dependencies, not the convert extras:

```bash
cd services/translate
pip install -r requirements.txt
TRANSLATE_BACKEND=ctranslate2 \
TRANSLATE_MODEL_DIR=models/opus-mt-en-zh-ct2 \
  uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1
```

The model directory must be readable by the process. `/health` stays 503 until `model.bin`, `source.spm`, and `target.spm` are present.

| variable | default | purpose |
| --- | --- | --- |
| `TRANSLATE_BACKEND` | `ctranslate2` | `ctranslate2` or `mock` |
| `TRANSLATE_MODEL_DIR` | `models/opus-mt-en-zh-ct2` | converted model directory |
| `TRANSLATE_GLOSSARY_DIR` | repo `glossary/` or `/glossary` | CSV directory |
| `TRANSLATE_DEFAULT_GLOSSARY` | `deep-learning` | course id |
| `TRANSLATE_INTRA_THREADS` | `2` | threads inside one translation |
| `TRANSLATE_INTER_THREADS` | `1` | parallel translations |
| `TRANSLATE_BEAM_SIZE` | `1` | raise for slightly slower, slightly better captions |
| `TRANSLATE_COMPUTE_TYPE` | from `runtime.json`, else `int8` | must match the conversion |
| `TRANSLATE_MAX_UNIT_CHARS` | `480` | sentence split for long text |
| `TRANSLATE_MAX_CHARS` | `20000` | request limit |
| `TRANSLATE_MAX_SEGMENTS` | `200` | request limit |

## Docker

Build from the repository root. The image does not contain weights.

```bash
docker build -f services/translate/Dockerfile -t translator-edu-translate:cpu .
docker compose -f services/translate/docker-compose.yml up --build
```

The compose file publishes host port **8081** and container port **8080**. Paste that service into `deploy/docker-compose.yml` when Eve's deploy file exists; the comment at the top of the compose file has the two path edits. From another container the base URL is `http://translate:8080`.

## Tests

CI runs pytest with `TRANSLATE_BACKEND=mock` and does not download weights. See `.github/workflows/translate.yml`.

```bash
cd services/translate
pip install -r requirements-ci.txt
pytest
```
