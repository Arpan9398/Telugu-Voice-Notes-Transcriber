"""Text normalisation and WER/CER computation (Jiwer)."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Sequence

_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff"}


def normalize_text(text: str | None) -> str:
    """NFC-normalise, casefold, drop zero-width chars, turn punctuation/symbols into spaces.

    Indic combining vowel signs (category M*) are kept; the Devanagari danda is
    punctuation (category P*) and is removed.
    """
    text = unicodedata.normalize("NFC", text or "").casefold()
    out = []
    for ch in text:
        if ch in _ZERO_WIDTH:
            continue
        out.append(" " if unicodedata.category(ch)[0] in ("P", "S") else ch)
    return " ".join("".join(out).split())


@dataclass(frozen=True)
class ErrorRates:
    wer: float
    cer: float
    ref_words: int
    ref_chars: int
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    char_errors: int


def compute_error_rates(refs: Sequence[str], hyps: Sequence[str]) -> ErrorRates:
    """Corpus-level WER and CER over aligned reference/hypothesis lists.

    Counts are summed over pairs, which equals Jiwer's corpus-level WER. A pair with
    an empty hypothesis counts as all-deletions. Pairs whose normalised reference is
    empty are skipped (WER is undefined for them).
    """
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps must have the same length.")
    import jiwer

    S = D = I = H = ref_words = 0
    ref_chars = char_err = 0
    used = 0
    for ref_raw, hyp_raw in zip(refs, hyps):
        ref, hyp = normalize_text(ref_raw), normalize_text(hyp_raw)
        if not ref:
            continue
        used += 1
        ref_words += len(ref.split())
        ref_chars += len(ref)
        if not hyp:
            D += len(ref.split())
            char_err += len(ref)
            continue
        w = jiwer.process_words(ref, hyp)
        S += w.substitutions
        D += w.deletions
        I += w.insertions
        H += w.hits
        c = jiwer.process_characters(ref, hyp)
        char_err += c.substitutions + c.deletions + c.insertions
    if used == 0:
        raise ValueError("No non-empty reference transcripts to score.")
    return ErrorRates(
        wer=(S + D + I) / ref_words,
        cer=char_err / ref_chars,
        ref_words=ref_words,
        ref_chars=ref_chars,
        substitutions=S,
        deletions=D,
        insertions=I,
        hits=H,
        char_errors=char_err,
    )
