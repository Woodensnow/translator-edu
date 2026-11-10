#!/usr/bin/env python3
"""Measure which glossary sentinels survive opus-mt-en-zh on CPU.

Not part of CI. Convert weights first with download_model.py, then:

    python scripts/benchmark_sentinels.py --write benchmarks/sentinel-results.md

The script does not download weights. It reports exact and case-insensitive
survival, how often the source sentencepiece marks a sentinel as <unk>, and
per-sentence latency with intra_threads=2 and beam size 1.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

from app.glossary import Match, restore_placeholders  # noqa: E402
DEFAULT_MODEL = SERVICE_ROOT / "models" / "opus-mt-en-zh-ct2"

# Fixed caption-sized lines. {0} and {1} are sentinel slots.
CASES = [
    "{0} uses gradient descent.",
    "A convolutional neural network applies {0}.",
    "The learning rate changes every {0}.",
    "{0} is the core of the {1}.",
    "{0} stabilizes training.",
    "The {0} updates the weights.",
    "{0} hurts generalization.",
    "Use a {0} for this sequence.",
    "The activation function is {0}.",
    "Mini-batch gradient descent uses a {0}.",
    "{0} and {1} are both regularizers.",
    "This {0} layer follows {1}.",
]
BASELINE = [
    "Backpropagation uses gradient descent.",
    "A convolutional neural network applies dropout.",
    "The learning rate changes every epoch.",
    "Attention is the core of the transformer.",
    "Batch normalization stabilizes training.",
    "The optimizer updates the weights.",
    "Overfitting hurts generalization.",
    "Use a recurrent network for this sequence.",
    "The activation function is ReLU.",
    "Mini-batch gradient descent uses a loss function.",
    "Dropout and weight decay are both regularizers.",
    "This convolution layer follows pooling.",
]
SPECIAL = {"<s>", "</s>", "<unk>", "<pad>"}


def schemes() -> dict[str, callable]:
    return {
        "brackets": lambda index: f"⟦G{index}⟧",
        "underscores": lambda index: f"__G{index}__",
        "digits": lambda index: f"{910000 + index}",
        "zxq": lambda index: f"ZXQ{index}",
        "zqx": lambda index: f"ZQX{index}",
    }


def fill(template: str, render) -> tuple[str, list[str]]:
    slots = []
    text = template
    for index in range(template.count("{")):
        token = render(index)
        slots.append(token)
        text = text.replace("{" + str(index) + "}", token, 1)
    return text, slots


def encode(processor, text: str) -> list[str]:
    pieces = list(processor.encode(text, out_type=str))
    if not pieces or pieces[-1] != "</s>":
        pieces.append("</s>")
    return pieces


def decode(processor, tokens: list[str]) -> str:
    kept = [token for token in tokens if token not in SPECIAL]
    if not kept:
        return ""
    return processor.decode(kept).strip()


def translate_one(translator, source_spm, target_spm, text: str) -> tuple[str, float, list[str]]:
    pieces = encode(source_spm, text)
    started = time.perf_counter()
    result = translator.translate_batch([pieces], beam_size=1, max_decoding_length=256)
    elapsed = time.perf_counter() - started
    hypothesis = result[0].hypotheses[0] if result and result[0].hypotheses else []
    return decode(target_spm, hypothesis), elapsed, pieces


def load_runtime(model_dir: Path, intra_threads: int):
    import ctranslate2
    import sentencepiece as spm

    source = spm.SentencePieceProcessor()
    target = spm.SentencePieceProcessor()
    source.Load(str(model_dir / "source.spm"))
    target.Load(str(model_dir / "target.spm"))
    translator = ctranslate2.Translator(
        str(model_dir),
        device="cpu",
        inter_threads=1,
        intra_threads=intra_threads,
        compute_type="int8",
    )
    return translator, source, target, getattr(ctranslate2, "__version__", "unknown")


def score_case(output: str, slots: list[str]) -> tuple[bool, bool, bool]:
    exact = all(slot in output for slot in slots)
    folded = output.casefold()
    insensitive = all(slot.casefold() in folded for slot in slots)
    if slots and all(slot.startswith("ZXQ") and slot[3:].isdigit() for slot in slots):
        matches = [Match(0, 0, slot, slot, "TERM", "", slot) for slot in slots]
        _restored, missing = restore_placeholders(output, matches)
        restored = not missing
    else:
        restored = exact
    return exact, insensitive, restored


def summarize(rows: list[dict]) -> dict:
    count = len(rows)
    return {
        "sentences": count,
        "exact": round(sum(row["exact"] for row in rows) / count, 4),
        "casefold": round(sum(row["casefold"] for row in rows) / count, 4),
        "restored": round(sum(row["restored"] for row in rows) / count, 4),
        "unk_rate": round(sum(row["unk"] for row in rows) / count, 4),
        "mean_ms": round(1000 * sum(row["seconds"] for row in rows) / count, 2),
        "max_ms": round(1000 * max(row["seconds"] for row in rows), 2),
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# Sentinel benchmark",
        "",
        "Measured with CTranslate2 int8 `Helsinki-NLP/opus-mt-en-zh` on CPU.",
        "Not run in CI. Regenerate with `python scripts/benchmark_sentinels.py --write benchmarks/sentinel-results.md`.",
        "",
        f"- ctranslate2: `{payload['ctranslate2']}`",
        f"- device: cpu, intra_threads={payload['intra_threads']}, inter_threads=1, beam=1, int8",
        f"- host: `{payload['host']}`",
        f"- chosen prefix: `{payload['chosen_prefix']}` ({payload['chosen_name']})",
        "",
        "Survival is the fraction of sentences whose every sentinel is still in the decoded text.",
        "`restored` uses the service rule: case-insensitive `ZXQ` digits, plus the observed `兹XQ` transliteration.",
        "`<unk>` is the fraction of source sentences where SentencePiece emitted `<unk>` for that line.",
        "Latency is one sentence per `translate_batch` call, after one warmup sentence.",
        "The winner is the best exact rate. `__G0__` is not used even when it copies sometimes: one line took hundreds of milliseconds.",
        "",
        "| scheme | example | exact | case-insensitive | restored | unk inputs | mean ms | max ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scheme in payload["schemes"]:
        lines.append(
            "| {name} | `{example}` | {exact:.0%} | {casefold:.0%} | {restored:.0%} | {unk_rate:.0%} | {mean_ms:.1f} | {max_ms:.1f} |".format(
                **scheme
            )
        )
    baseline = payload["baseline"]
    lines.extend(
        [
            "",
            f"Plain sentences, no sentinel: mean {baseline['mean_ms']:.1f} ms, max {baseline['max_ms']:.1f} ms.",
            "",
            "## Per-sentence latency",
            "",
            "Same twelve lines. Baseline is ordinary English. Chosen is that line with the winning sentinel.",
            "",
            "| sentence | baseline ms | chosen ms | chosen survived |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for row in payload["per_sentence"]:
        survived = "yes" if row["survived"] else "no"
        lines.append(
            f"| {row['label']} | {row['baseline_ms']:.1f} | {row['chosen_ms']:.1f} | {survived} |"
        )
    if payload["failures"]:
        lines.extend(["", "## Samples that lost the sentinel", ""])
        for failure in payload["failures"]:
            lines.append(f"- `{failure['scheme']}` `{failure['input']}` -> `{failure['output']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--intra-threads", type=int, default=2)
    parser.add_argument("--write", type=Path, default=None, help="markdown results path; JSON is written beside it")
    args = parser.parse_args()
    model_dir = args.model
    required = ["model.bin", "source.spm", "target.spm"]
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise SystemExit(f"missing {missing} under {model_dir}. Run scripts/download_model.py first.")

    translator, source_spm, target_spm, ct2_version = load_runtime(model_dir, args.intra_threads)
    # Warm the weights before timing.
    translate_one(translator, source_spm, target_spm, "Hello from the lecture.")

    scheme_payloads = []
    failure_samples = []
    detailed: dict[str, list[dict]] = {}
    for name, render in schemes().items():
        rows = []
        for template in CASES:
            text, slots = fill(template, render)
            output, elapsed, pieces = translate_one(translator, source_spm, target_spm, text)
            exact, folded, restored = score_case(output, slots)
            rows.append(
                {
                    "input": text,
                    "output": output,
                    "seconds": elapsed,
                    "exact": exact,
                    "casefold": folded,
                    "restored": restored,
                    "unk": "<unk>" in pieces,
                    "slots": slots,
                }
            )
            if not restored and sum(1 for item in failure_samples if item["scheme"] == name) < 2:
                failure_samples.append({"scheme": name, "input": text, "output": output})
        summary = summarize(rows)
        summary["name"] = name
        summary["example"] = render(0)
        scheme_payloads.append(summary)
        detailed[name] = rows

    baseline_rows = []
    for text in BASELINE:
        _output, elapsed, _pieces = translate_one(translator, source_spm, target_spm, text)
        baseline_rows.append({"input": text, "seconds": elapsed})
    baseline = {
        "mean_ms": round(1000 * sum(row["seconds"] for row in baseline_rows) / len(baseline_rows), 2),
        "max_ms": round(1000 * max(row["seconds"] for row in baseline_rows), 2),
    }

    # Prefer exact copies. Case-insensitive survival is the tie break, then speed.
    winner = max(scheme_payloads, key=lambda item: (item["exact"], item["casefold"], -item["mean_ms"]))
    chosen_rows = detailed[winner["name"]]
    per_sentence = []
    for index, row in enumerate(chosen_rows):
        per_sentence.append(
            {
                "label": CASES[index].replace("{0}", "TERM").replace("{1}", "TERM"),
                "baseline_ms": round(1000 * baseline_rows[index]["seconds"], 2),
                "chosen_ms": round(1000 * row["seconds"], 2),
                "survived": row["restored"],
            }
        )

    prefix = {"brackets": "⟦G", "underscores": "__G", "digits": "910000", "zxq": "ZXQ", "zqx": "ZQX"}[winner["name"]]
    payload = {
        "ctranslate2": ct2_version,
        "intra_threads": args.intra_threads,
        "host": platform.platform(),
        "chosen_name": winner["name"],
        "chosen_prefix": prefix,
        "schemes": scheme_payloads,
        "baseline": baseline,
        "per_sentence": per_sentence,
        "failures": failure_samples,
    }
    print(render_markdown(payload))
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(render_markdown(payload), encoding="utf-8")
        json_path = args.write.with_suffix(".json")
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.write} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
