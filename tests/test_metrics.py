import pytest

from tvt.metrics import normalize_text


def test_normalize_strips_punctuation_and_danda():
    assert normalize_text("नमस्ते, आप कैसे हैं?") == "नमस्ते आप कैसे हैं"
    assert normalize_text("आज मौसम बहुत अच्छा है।") == "आज मौसम बहुत अच्छा है"
    assert normalize_text("ఈ రోజు వాతావరణం చాలా బాగుంది.") == "ఈ రోజు వాతావరణం చాలా బాగుంది"


def test_normalize_handles_none_and_whitespace():
    assert normalize_text(None) == ""
    assert normalize_text("  a \n b  ") == "a b"


def test_normalize_keeps_telugu_vowel_signs():
    assert "ా" in normalize_text("చాలా")


def test_wer_known_values():
    pytest.importorskip("jiwer")
    from tvt.metrics import compute_error_rates

    r = compute_error_rates(["the cat sat"], ["the cat sat"])
    assert r.wer == 0 and r.cer == 0
    r = compute_error_rates(["the cat sat"], ["the dog sat"])
    assert r.wer == pytest.approx(1 / 3) and r.substitutions == 1
    r = compute_error_rates(["the cat sat"], [""])
    assert r.wer == 1.0 and r.deletions == 3
    r = compute_error_rates(["a b"], ["a b c d"])
    assert r.wer == 1.0 and r.insertions == 2


def test_wer_ignores_punctuation_and_empty_refs():
    pytest.importorskip("jiwer")
    from tvt.metrics import compute_error_rates

    assert compute_error_rates(["నమస్కారం, మీరు?"], ["నమస్కారం మీరు"]).wer == 0
    with pytest.raises(ValueError):
        compute_error_rates([""], ["x"])
