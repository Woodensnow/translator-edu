# Sentinel benchmark

Measured with CTranslate2 int8 `Helsinki-NLP/opus-mt-en-zh` on CPU.
Not run in CI. Regenerate with `python scripts/benchmark_sentinels.py --write benchmarks/sentinel-results.md`.

These figures are twelve synthetic sentences on a dev machine with `intra_threads=2`, not on Tim's 2-core Alibaba server. Re-run this script there with 50 or more real lecture sentences before treating the latencies as that machine's budget, and record how often the fragment fallback runs.

- ctranslate2: `4.8.5`
- device: cpu, intra_threads=2, inter_threads=1, beam=1, int8
- host: `Linux-6.12.113+-x86_64-with-glibc2.39`
- chosen prefix: `ZXQ` (zxq)

Survival is the fraction of sentences whose every sentinel is still in the decoded text.
`restored` uses the service rule: case-insensitive `ZXQ` digits, plus the observed `兹XQ` transliteration.
`<unk>` is the fraction of source sentences where SentencePiece emitted `<unk>` for that line.
Latency is one sentence per `translate_batch` call, after one warmup sentence.
The winner is the best exact rate. `__G0__` is not used even when it copies sometimes: one line took hundreds of milliseconds.

| scheme | example | exact | case-insensitive | restored | unk inputs | mean ms | max ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| brackets | `⟦G0⟧` | 0% | 0% | 0% | 0% | 32.4 | 69.9 |
| underscores | `__G0__` | 33% | 33% | 33% | 0% | 121.9 | 940.0 |
| digits | `910000` | 0% | 0% | 0% | 0% | 29.4 | 42.3 |
| zxq | `ZXQ0` | 92% | 92% | 100% | 0% | 39.8 | 50.9 |
| zqx | `ZQX0` | 0% | 0% | 0% | 0% | 33.6 | 50.8 |

Plain sentences, no sentinel: mean 33.0 ms, max 45.2 ms.

## Per-sentence latency

Same twelve lines. Baseline is ordinary English. Chosen is that line with the winning sentinel.

| sentence | baseline ms | chosen ms | chosen survived |
| --- | ---: | ---: | --- |
| TERM uses gradient descent. | 34.7 | 33.1 | yes |
| A convolutional neural network applies TERM. | 38.5 | 39.4 | yes |
| The learning rate changes every TERM. | 27.9 | 50.2 | yes |
| TERM is the core of the TERM. | 26.1 | 39.5 | yes |
| TERM stabilizes training. | 28.6 | 25.8 | yes |
| The TERM updates the weights. | 26.0 | 34.8 | yes |
| TERM hurts generalization. | 28.7 | 35.6 | yes |
| Use a TERM for this sequence. | 29.4 | 34.0 | yes |
| The activation function is TERM. | 29.8 | 36.4 | yes |
| Mini-batch gradient descent uses a TERM. | 44.5 | 50.9 | yes |
| TERM and TERM are both regularizers. | 36.7 | 48.0 | yes |
| This TERM layer follows TERM. | 45.2 | 49.7 | yes |

## Samples that lost the sentinel

- `brackets` `⟦G0⟧ uses gradient descent.` -> `使用梯度下沉。`
- `brackets` `A convolutional neural network applies ⟦G0⟧.` -> `神经网络在变幻中应用“G0”`
- `underscores` `A convolutional neural network applies __G0__.` -> `进化神经网络应用_G0_______________________。`
- `underscores` `__G0__ is the core of the __G1__.` -> `G0__是G1__的核心。`
- `digits` `910000 uses gradient descent.` -> `91000使用梯度下降法。`
- `digits` `A convolutional neural network applies 910000.` -> `神经网络的变速率是91000`
- `zqx` `ZQX0 uses gradient descent.` -> `X0使用梯度下降。`
- `zqx` `A convolutional neural network applies ZQX0.` -> `神经神经网络的变速应用是X0`
