# Saved results

Completed upload jobs are written under `RESULT_DIR` (default `/data/results`). Each file is one JSON document: the lecture name, the time it finished, the duration, and the bilingual segments. `.txt` and `.srt` are rendered from those segments when someone downloads them. The audio temp file is deleted as soon as processing finishes and is not copied here.

The page keeps only the latest few jobs in memory, same as before. Opening an older result reads the file. The files are still there after the web container restarts.

## How long they stay

`RESULT_RETENTION_DAYS` defaults to 7. A result younger than that is kept. Older files are deleted when the web process starts and about once an hour after that. `0` means keep them until someone presses delete.

The write is a temporary file in the same directory, then `rename`. A crash mid-write does not leave a half JSON file under the job id. Job ids are twelve lowercase hex characters. A path such as `../` is rejected and is never joined onto `RESULT_DIR`.

## Page and API

Every route below uses the same `WEB_ACCESS_TOKEN` as the rest of the captions API.

| Method | Path | What it does |
| --- | --- | --- |
| `GET` | `/results` | List. Each row is `id`, `name`, `finished_at`, `duration`. Segment text is not included. |
| `GET` | `/results/{id}` | One result, including segments, so the page can open it again. |
| `GET` | `/results/{id}.txt` | Bilingual text download. |
| `GET` | `/results/{id}.srt` | Bilingual SubRip download. |
| `DELETE` | `/results/{id}` | Remove that file. The page asks for confirmation first. |

The volume, the uid that must be able to write it, and the bind-mount alternative are in [deploy.md](deploy.md).
