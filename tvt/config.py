"""Central configuration (environment variables are read once at import)."""
from __future__ import annotations

import os
from pathlib import Path

try:  # .env is optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent

MODEL_CHOICES = ("base", "small")

# UI label -> Whisper language code (None = let Whisper auto-detect)
LANGUAGES: dict[str, str | None] = {
    "Telugu (te)": "te",
    "Hindi (hi)": "hi",
    "Auto-detect": None,
}

TARGET_SR = 16_000  # Whisper expects 16 kHz mono

MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "600"))
MIN_AUDIO_SECONDS = float(os.getenv("MIN_AUDIO_SECONDS", "0.5"))
SILENCE_RMS_THRESHOLD = float(os.getenv("SILENCE_RMS_THRESHOLD", "0.001"))

WHISPER_MODEL_DIR = os.getenv("WHISPER_MODEL_DIR") or None
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE") or None

EVAL_MANIFEST = ROOT / "data" / "eval" / "manifest.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_JSON = RESULTS_DIR / "wer_results.json"
RESULTS_CSV = RESULTS_DIR / "wer_per_clip.csv"
