import numpy as np
import pytest

from tvt.audio import AudioValidationError, load_audio_bytes, load_audio_file, validate_samples

SR = 16_000


def tone(seconds=1.0, amp=0.3):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_valid_audio_passes():
    clip = validate_samples(tone(), SR)
    assert clip.duration == pytest.approx(1.0)


def test_empty_samples_rejected():
    with pytest.raises(AudioValidationError, match="empty"):
        validate_samples(np.array([], dtype=np.float32), SR)


def test_silence_rejected():
    with pytest.raises(AudioValidationError, match="silent"):
        validate_samples(np.zeros(SR * 2, dtype=np.float32), SR)


def test_too_short_rejected():
    with pytest.raises(AudioValidationError, match="too short"):
        validate_samples(tone(0.1), SR)


def test_nan_rejected():
    x = tone()
    x[10] = np.nan
    with pytest.raises(AudioValidationError, match="invalid"):
        validate_samples(x, SR)


def test_empty_bytes_rejected():
    with pytest.raises(AudioValidationError):
        load_audio_bytes(b"", "x.wav")
    with pytest.raises(AudioValidationError):
        load_audio_bytes(None, "x.wav")


def test_missing_file_rejected(tmp_path):
    with pytest.raises(AudioValidationError, match="not found"):
        load_audio_file(tmp_path / "nope.wav")


def test_garbage_bytes_rejected():
    pytest.importorskip("librosa")
    with pytest.raises(AudioValidationError, match="decode"):
        load_audio_bytes(b"this is not audio" * 50, "x.wav")


def test_wav_roundtrip(tmp_path):
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("librosa")
    p = tmp_path / "t.wav"
    sf.write(p, tone(1.5), SR)
    assert load_audio_file(p).duration == pytest.approx(1.5, abs=0.05)
