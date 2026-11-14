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

Extra columns are ignored. `en` and `zh` are required on every row. Matching prefers the longest phrase. A hyphen in the English term also matches a space or nothing (`fine-tuning`, `fine tuning`, `finetuning`). Ordinary terms accept a trailing `s` or `es` (`weights`, `mini-batches`, `biases`). All-caps acronyms match only as written, plus a lowercase plural `s` (`AI`, `AIs`, `CNNs`), so `ais` is not treated as AI. A title-case name such as `Adam` has no plural, so `Adams` is left alone. Spaced and closed spellings of a hyphenated term do not need their own rows.

The starter course is Deep Learning and Neural Networks: `glossary/deep-learning.csv`. Wording follows 《动手学深度学习》 where that book has a settled translation, including 暂退法, 批量规范化, 汇聚, and 迭代周期. A few names stay in Latin (`Transformer`, `Softmax`, `ReLU`, `Adam`, and model or optimizer names such as `BERT`, `GPT`, `RMSProp`).

The ASR service reads this file as the Whisper prompt and truncates it by character count (`ASR_HOTWORD_MAX_CHARS` / `ASR_PROMPT_MAX_CHARS`). That limit can be raised at deploy time. Put the most important terms first: the ones most likely to be spoken and most likely to be misheard, such as multi-word technical phrases, acronyms, and names (`convolutional neural network`, `CNN`, `LSTM`, `ReLU`, `Adam`, `back-propagation`), ahead of common words Whisper already spells (`weight`, `gradient`, `parameter`). Every row is still a translation term.

## Add a term

1. Open the course CSV, or copy it to `glossary/<course-id>.csv` for a new course.
2. Add a row. Prefer a specific phrase (`loss function`) over a short ambiguous word (`loss`). Put a term the ASR prompt should include near the top, because the prompt is cut off by character count.
3. Keep the file UTF-8. Quote the note if it contains a comma.
4. The service reloads when the file changes; a broken save keeps the previous good list.

Call the API with `"glossary": "<course-id>"`. The default course is `deep-learning`.
