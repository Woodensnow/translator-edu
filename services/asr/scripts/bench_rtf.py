#!/usr/bin/env python3
"""Measure real-time factor for the configured Whisper model.

Pins the process to two CPUs when the OS allows it, so the number is comparable
to a 2-core server. Does not run in CI.

Example::

    python scripts/bench_rtf.py --audio /tmp/bench-audio/jfk.flac --repeat 3
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr.audio import decode_audio_bytes  # noqa: E402
from asr.engine import WhisperEngine  # noqa: E402
from asr.glossary import build_hotwords, build_initial_prompt, load_english_terms  # noqa: E402
from asr.settings import Settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure ASR real-time factor on CPU")
    parser.add_argument("--audio", required=True, help="WAV, flac, or another file PyAV can open")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the clip to get a longer sample")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    if hasattr(os, "sched_setaffinity"):
        cores = set(range(min(args.threads, os.cpu_count() or args.threads)))
        os.sched_setaffinity(0, cores)

    overrides = dict(os.environ)
    overrides["ASR_CPU_THREADS"] = str(args.threads)
    settings = Settings.from_env(overrides)
    audio_path = Path(args.audio)
    samples, sample_rate = decode_audio_bytes(audio_path.read_bytes(), audio_path.name)
    if args.repeat > 1:
        import numpy as np

        samples = np.tile(samples, args.repeat)
    terms = []
    if settings.glossary_path.is_file():
        terms = load_english_terms(settings.glossary_path)
    prompt = build_initial_prompt(terms)
    hotwords = build_hotwords(terms, max_chars=settings.hotword_max_chars)

    engine = WhisperEngine(settings)
    warmup = samples[: sample_rate]
    engine.transcribe(warmup, sample_rate, prompt, hotwords, vad=False)
    started = time.perf_counter()
    segments = engine.transcribe(samples, sample_rate, prompt, hotwords, vad=False)
    elapsed = time.perf_counter() - started
    audio_s = float(samples.size) / float(sample_rate)
    preview = segments[0].text if segments else ""
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "model": settings.model,
                "compute_type": settings.compute_type,
                "cpu_threads": settings.cpu_threads,
                "audio_s": round(audio_s, 3),
                "elapsed_s": round(elapsed, 3),
                "rtf": round(elapsed / audio_s, 3) if audio_s else None,
                "rss_mb": round(rss_kb / 1024, 1),
                "segments": len(segments),
                "preview": preview[:160],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
