# Course glossaries

Shared term lists for lecture transcription and translation. One UTF-8 CSV per course. The translation service locks these terms so they stay in mainland classroom wording. The ASR service can read the same file as prompt terms; there is no separate export.

## Format

`glossary/<course-id>.csv` with this header:

```csv
en,zh,note
```

| column | required | meaning |
| --- | --- | --- |
| `en` | yes | English term as spoken in class |
| `zh` | yes | Mainland Chinese rendering to force |
| `note` | yes, may be empty | Hint for lecturers. Not sent to the model |

Extra columns are ignored. `en` and `zh` are required on every row. Matching is case-insensitive, prefers the longest phrase, and accepts a trailing plural `s` (`weights`, `CNNs`). A term that already ends in `s` also accepts `-es` (`bias` / `biases`).

The starter course is Deep Learning and Neural Networks: `glossary/deep-learning.csv`. Wording follows 《动手学深度学习》 where that book has a settled translation, including 暂退法, 批量规范化, 汇聚, and 迭代周期. A few names stay in Latin (`Transformer`, `Softmax`, `ReLU`, `Adam`).

## Add a term

1. Open the course CSV, or copy it to `glossary/<course-id>.csv` for a new course.
2. Add a row. Prefer a specific phrase (`loss function`) over a short ambiguous word (`loss`).
3. Keep the file UTF-8. Quote the note if it contains a comma.
4. The service reloads when the file changes; a broken save keeps the previous good list.

Call the API with `"glossary": "<course-id>"`. The default course is `deep-learning`.
