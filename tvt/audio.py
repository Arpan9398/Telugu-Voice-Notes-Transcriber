"""Audio loading and validation (librosa is imported lazily)."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config


class AudioValidationError(ValueError):
    """Audio cannot be transcribed (empty, corrupt, too short, silent, too long)."""


@dataclass(frozen=True)
class AudioClip:
    samples: np.ndarray  # float32, mono, 16 kHz
    sample_rate: int
    duration: float  # seconds
    rms: float


def validate_samples(samples: np.ndarray, sample_rate: int) -> AudioClip:
    """Validate decoded samples and return an AudioClip, or raise AudioValidationError."""
    samples = np.asarray(samples)
    if samples.size == 0:
        raise AudioValidationError("The audio contains no samples (empty recording).")
    samples = samples.astype(np.float32, copy=False)
    if not np.all(np.isfinite(samples)):
        raise AudioValidationError("The audio contains invalid (NaN/Inf) values and looks corrupt.")

    duration = samples.size / float(sample_rate)
    if duration < config.MIN_AUDIO_SECONDS:
        raise AudioValidationError(
            f"The audio is too short ({duration:.2f}s). Minimum is {config.MIN_AUDIO_SECONDS:g}s."
        )
    if duration > config.MAX_AUDIO_SECONDS:
        raise AudioValidationError(
            f"The audio is too long ({duration / 60:.1f} min). "
            f"Maximum is {config.MAX_AUDIO_SECONDS / 60:.1f} min - please trim it."
        )

    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if peak < 1e-4 or rms < config.SILENCE_RMS_THRESHOLD:
        raise AudioValidationError(
            "The audio appears to be silent. Check your microphone or the file and try again."
        )
    return AudioClip(samples=samples, sample_rate=sample_rate, duration=duration, rms=rms)


def load_audio_file(path: str | os.PathLike) -> AudioClip:
    """Decode an audio file to 16 kHz mono float32 and validate it."""
    path = Path(path)
    if not path.is_file():
        raise AudioValidationError(f"Audio file not found: {path}")
    if path.stat().st_size == 0:
        raise AudioValidationError("The audio file is empty (0 bytes).")

    import librosa  # lazy: heavy import

    try:
        samples, sr = librosa.load(str(path), sr=config.TARGET_SR, mono=True)
    except Exception as exc:  # decoder errors vary by backend
        raise AudioValidationError(
            "Could not decode this audio. The file may be corrupt or in an unsupported "
            "format (installing ffmpeg enables more formats, e.g. m4a/webm)."
        ) from exc
    return validate_samples(samples, sr)


def load_audio_bytes(data: bytes | None, filename: str = "audio.wav") -> AudioClip:
    """Decode uploaded/recorded bytes. `filename` supplies the extension hint."""
    if not data:
        raise AudioValidationError("No audio data received (empty upload or recording).")
    suffix = Path(filename).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        return load_audio_file(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
