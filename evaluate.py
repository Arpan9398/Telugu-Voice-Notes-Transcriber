#!/usr/bin/env python3
"""Evaluate Whisper models on the sample manifest and print WER/CER.

    python evaluate.py                     # base + small on data/eval/manifest.csv
    python evaluate.py --models base       # one model
    python evaluate.py --manifest my.csv   # your own clips
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tvt import config
from tvt.evaluation import evaluate_models, save_report
from tvt.transcriber import TranscriptionError


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=config.EVAL_MANIFEST)
    p.add_argument("--models", nargs="+", default=list(config.MODEL_CHOICES), choices=config.MODEL_CHOICES)
    p.add_argument("--json-out", type=Path, default=config.RESULTS_JSON)
    p.add_argument("--csv-out", type=Path, default=config.RESULTS_CSV)
    args = p.parse_args()

    def progress(done: int, total: int, msg: str) -> None:
        print(f"[{done}/{total}] {msg}", file=sys.stderr)

    try:
        report = evaluate_models(args.models, args.manifest, progress)
    except (FileNotFoundError, ValueError, TranscriptionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    for s in report.skipped:
        print(f"SKIPPED {s['id']}: {s['reason']}", file=sys.stderr)
    for f in report.failed:
        print(f"FAILED  {f['id']} ({f['model']}): {f['reason']}", file=sys.stderr)

    if not report.per_clip:
        print(
            "\nNo clips were evaluated. The sample audio is not bundled: run\n"
            "    python scripts/generate_sample_audio.py\n"
            "first (needs internet), or add your own clips to the manifest.",
            file=sys.stderr,
        )
        return 1

    save_report(report, args.json_out, args.csv_out)

    print(f"\nWhisper {report.meta['openai_whisper_version']} on {report.meta['device']} | "
          f"{report.meta['clips_evaluated']}/{report.meta['clips_in_manifest']} clips evaluated\n")
    header = f"{'model':<7}{'lang':<6}{'clips':>6}{'words':>7}{'WER %':>9}{'CER %':>9}{'S':>5}{'D':>5}{'I':>5}{'s/clip':>8}"
    print(header)
    print("-" * len(header))
    for s in report.summary:
        print(f"{s['model']:<7}{s['language']:<6}{s['clips']:>6}{s['ref_words']:>7}"
              f"{s['wer'] * 100:>9.1f}{s['cer'] * 100:>9.1f}{s['substitutions']:>5}"
              f"{s['deletions']:>5}{s['insertions']:>5}{s['avg_seconds']:>8.2f}")
    print(f"\nSaved: {args.json_out}\n       {args.csv_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
