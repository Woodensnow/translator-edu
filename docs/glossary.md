# Glossary file the ASR reads

Course lists live in `glossary/<course>.csv`. The translate owner documents the columns, matching rules, and how to add a row in [glossary/README.md](../glossary/README.md). The starter course is `glossary/deep-learning.csv` with header `en,zh,note`.

ASR does not write that file. It only reads the English column, as a Whisper initial prompt and as faster-whisper hotwords. Chinese text and the notes column are never copied into the prompt.

## Path

`ASR_GLOSSARY_PATH` defaults to `glossary/deep-learning.csv`. In Compose it is `/glossary/deep-learning.csv`, from a read-only mount of the repo `glossary/` directory. The same mount is used by translate (`TRANSLATE_GLOSSARY_DIR=/glossary`) so an edit on the host is the file both processes see. ASR reloads when the file's modification time changes. Translate reloads on its own; that behavior is in the translate service, not here.

If the file is missing, ASR logs a warning and transcribes with no terms. A header it does not recognize stops the process.

The loader is `services/asr/src/asr/glossary.py`. It accepts the header `en,zh,note` (any English capitalization) and also `英文,中文,备注`. A missing notes column is treated as empty. The committed course file uses `en,zh,note`.

Tests use `services/asr/tests/fixtures/deep-learning.csv` (`backpropagation`, `gradient descent`, `activation function`). They do not read `glossary/deep-learning.csv`.
