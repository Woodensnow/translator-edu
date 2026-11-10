# Captions to translation

The translate HTTP API belongs to `services/translate`. Its request, response, status codes, and glossary routes are specified in [services/translate/README.md](../services/translate/README.md). This page only describes how ASR and the captions page use that API. It is not a second contract.

## Path

1. The browser streams microphone PCM to the captions page at `GET /ws/transcribe`. The page tunnels that socket to ASR (`ASR_WS_URL`, default `ws://asr:8001/ws/transcribe`).
2. ASR sends final segments `{id, start, end, text}`. `start` and `end` are seconds. Ids look like `seg-1`.
3. The browser posts those segments to the captions page. The page forwards them as `POST /translate` to `TRANSLATE_URL` (default `http://translate:8080`). It does not rename fields.
4. The page reads `segments[].translation` and draws it next to the English line with the same `id`. Timestamps on the way back are the ones ASR sent.

A recording upload is `POST /jobs` on the captions page. The page streams the file to disk and asks ASR for one window at a time (`POST /uploads`, then `POST /uploads/{id}/next`). Each window's segments go through the same `POST /translate` step, and the page polls `GET /jobs/{id}?after=<n>` for the new rows.

Example body the page sends (the response shape, including `translation`, `segments[].translation`, `segments[].terms`, and statuses 422, 404, and 503, is in the translate README):

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

`glossary` is the course id (`COURSE_ID`, default `deep-learning`), matching `glossary/<id>.csv`.

## Small requests

Live mode sends one segment per request. A finished recording is packed into batches of at most 4 segments and at most 1500 characters of text. One over-long line is clipped to 1800 characters before it is sent. That stays well under 2000 characters.

## When Chinese is missing

If translate returns 503, or the forward times out (the page turns that into 504), the Chinese column shows the English sentence and the note "中文暂不可用". The English column is unchanged. The same fallback is used when the translate host cannot be reached.

`GET /health` and `GET /glossary` are for operators on the private network. The browser does not call them. Translate's own README is the description of those routes.
