#!/usr/bin/env python3
"""Download Helsinki-NLP/opus-mt-en-zh and convert it to CTranslate2 for CPU inference.

The runtime image does not include PyTorch. Run this once on a machine with a
few GB of RAM, then copy the output directory to the server. Do not commit it.

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements-convert.txt
    python scripts/download_model.py
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Helsinki-NLP/opus-mt-en-zh"
COPY_FILES = ["source.spm", "target.spm", "vocab.json", "tokenizer_config.json"]
QUANTIZATION_TO_COMPUTE = {
    "int8": "int8",
    "int8_float32": "int8_float32",
    "float32": "float32",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Helsinki-NLP/opus-mt-en-zh to a CPU CTranslate2 model.",
        epilog=(
            "Install a CPU PyTorch wheel before the convert extras:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  pip install -r requirements-convert.txt\n"
            "Weights are not committed. The server only needs the converted directory.\n"
            "float16 is omitted on purpose: these classroom servers have no GPU, and "
            "int8 is the CTranslate2 CPU path."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    parser.add_argument(
        "--output",
        type=Path,
        default=SERVICE_ROOT / "models" / "opus-mt-en-zh-ct2",
        help="directory for the converted model (gitignored)",
    )
    parser.add_argument(
        "--quantization",
        default="int8",
        choices=sorted(QUANTIZATION_TO_COMPUTE),
        help="int8 is the default for a 2-core CPU server",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output: Path = args.output
    if output.exists() and not output.is_dir():
        print(f"{output} exists and is not a directory.", file=sys.stderr)
        return 1
    if output.is_dir() and any(output.iterdir()) and not args.force:
        print(f"{output} already exists. Pass --force to replace it.", file=sys.stderr)
        return 1
    try:
        import ctranslate2
    except ImportError:
        print(
            "Missing convert dependencies. Install a CPU PyTorch wheel, then "
            "pip install -r requirements-convert.txt",
            file=sys.stderr,
        )
        return 1

    if output.exists() and args.force:
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Converting {args.model} to {output} ({args.quantization}).")
    print("This downloads the fp32 checkpoint (about 300MB) and needs a few GB of RAM.")
    converter_kwargs: dict[str, object] = {"copy_files": COPY_FILES}
    parameters = inspect.signature(ctranslate2.converters.TransformersConverter).parameters
    if "low_cpu_mem_usage" in parameters:
        converter_kwargs["low_cpu_mem_usage"] = True
    converter = ctranslate2.converters.TransformersConverter(args.model, **converter_kwargs)
    converter.convert(str(output), quantization=args.quantization, force=True)

    missing = [
        name for name in ("model.bin", "source.spm", "target.spm") if not (output / name).is_file()
    ]
    if missing:
        print(f"Conversion finished but missing {', '.join(missing)}.", file=sys.stderr)
        return 1

    runtime = {
        "model": args.model,
        "quantization": args.quantization,
        "compute_type": QUANTIZATION_TO_COMPUTE[args.quantization],
        "runtime": "ctranslate2",
        "device": "cpu",
    }
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(f"Wrote {output} ({size / 1_000_000:.1f} MB).")
    print("Do not commit this directory. Point TRANSLATE_MODEL_DIR at it and run the service.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
