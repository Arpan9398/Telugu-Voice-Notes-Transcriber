"""Whisper wrapper with a process-wide model cache."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import config


class TranscriptionError(RuntimeError):
    """Model could not be loaded or transcription failed."""


@dataclass
class TranscriptionResult:
    model: str
    text: str
    language: str  # language Whisper used / detected
    seconds: float  # wall-clock inference time
    segments: list[dict] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


_MODELS: dict[str, object] = {}
_LOCK = threading.Lock()


def get_device() -> str:
    if config.WHISPER_DEVICE:
        return config.WHISPER_DEVICE
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_model(name: str):
    """Load (and cache) a Whisper model. Weights are downloaded on first use."""
    if name not in config.MODEL_CHOICES:
        raise TranscriptionError(f"Unsupported model '{name}'. Choose from {config.MODEL_CHOICES}.")
    with _LOCK:
        if name in _MODELS:
            return _MODELS[name]
        try:
            import whisper
        except ImportError as exc:
            raise TranscriptionError(
                "openai-whisper is not installed. Run: pip install -r requirements.txt"
            ) from exc
        try:
            _MODELS[name] = whisper.load_model(
                name, device=get_device(), download_root=config.WHISPER_MODEL_DIR
            )
        except Exception as exc:
            raise TranscriptionError(
                f"Could not load Whisper '{name}' (first run needs internet to download "
                f"the weights): {exc}"
            ) from exc
        return _MODELS[name]


def transcribe_audio(
    samples: np.ndarray, model_name: str, language: str | None = None
) -> TranscriptionResult:
    """Transcribe 16 kHz mono float32 samples. language=None -> auto-detect."""
    model = load_model(model_name)
    try:
        start = time.perf_counter()
        result = model.transcribe(
            samples.astype(np.float32, copy=False),
            language=language,
            task="transcribe",
            fp16=(get_device() == "cuda"),
            temperature=0.0,
            condition_on_previous_text=False,
            verbose=None,
        )
        elapsed = time.perf_counter() - start
    except Exception as exc:
        raise TranscriptionError(f"Whisper '{model_name}' failed: {exc}") from exc

    return TranscriptionResult(
        model=model_name,
        text=(result.get("text") or "").strip(),
        language=result.get("language") or (language or "unknown"),
        seconds=elapsed,
        segments=[
            {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in result.get("segments", [])
        ],
    )
