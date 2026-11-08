#!/usr/bin/env python3
"""Create sample clips from the reference sentences in data/eval/manifest.csv using Google TTS.

Requires internet access (gTTS calls Google's service). The clips are SYNTHETIC speech,
so WER on them is NOT representative of real voice notes - for a meaningful evaluation,
record your own clips and add them to the manifest.

    python scripts/generate_sample_audio.py [--force]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tvt import config  # noqa: E402
from tvt.evaluation import load_manifest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=config.EVAL_MANIFEST)
    ap.add_argument("--force", action="store_true", help="regenerate existing clips")
    args = ap.parse_args()

    try:
        import librosa
        import soundfile as sf
        from gtts import gTTS
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2

    made = skipped = failed = 0
    for row in load_manifest(args.manifest):
        out = args.manifest.parent / row["audio"]
        if out.exists() and not args.force:
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                mp3 = Path(tmp) / "clip.mp3"
                gTTS(text=row["reference"], lang=row["language"]).save(str(mp3))
                samples, sr = librosa.load(str(mp3), sr=config.TARGET_SR, mono=True)
            sf.write(str(out), samples, sr)
            made += 1
            print(f"created {out.name}")
        except Exception as exc:
            failed += 1
            print(f"FAILED {row['id']}: {exc}", file=sys.stderr)
    print(f"done: {made} created, {skipped} already present, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
