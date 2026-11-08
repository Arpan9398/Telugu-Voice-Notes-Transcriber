"""Run Whisper models over a manifest of (audio, reference) pairs and score them."""
from __future__ import annotations

import csv
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from . import config
from .audio import AudioClip, AudioValidationError, load_audio_file
from .metrics import compute_error_rates
from .transcriber import TranscriptionError, get_device, load_model, transcribe_audio

ProgressFn = Callable[[int, int, str], None]


@dataclass
class EvaluationReport:
    meta: dict
    summary: list[dict]
    per_clip: list[dict]
    skipped: list[dict] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "meta": self.meta,
            "summary": self.summary,
            "per_clip": self.per_clip,
            "skipped": self.skipped,
            "failed": self.failed,
        }


def load_manifest(path: Path = config.EVAL_MANIFEST) -> list[dict]:
    """Manifest columns: id, audio (relative to manifest), language, reference."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    required = {"id", "audio", "language", "reference"}
    if not rows or not required.issubset(rows[0].keys()):
        raise ValueError(f"Manifest must have columns {sorted(required)} and at least one row.")
    return rows


def _whisper_version() -> str:
    try:
        from importlib.metadata import version

        return version("openai-whisper")
    except Exception:
        return "unknown"


def _summarise(rows: list[dict], model: str, language: str) -> dict | None:
    sel = [r for r in rows if r["model"] == model and (language == "all" or r["language"] == language)]
    if not sel:
        return None
    rates = compute_error_rates([r["reference"] for r in sel], [r["hypothesis"] for r in sel])
    return {
        "model": model,
        "language": language,
        "clips": len(sel),
        "ref_words": rates.ref_words,
        "wer": rates.wer,
        "cer": rates.cer,
        "substitutions": rates.substitutions,
        "deletions": rates.deletions,
        "insertions": rates.insertions,
        "avg_seconds": sum(r["seconds"] for r in sel) / len(sel),
    }


def evaluate_models(
    models: Sequence[str],
    manifest_path: Path = config.EVAL_MANIFEST,
    progress: ProgressFn | None = None,
) -> EvaluationReport:
    """Transcribe every usable manifest clip with each model and compute WER/CER.

    Clips whose audio is missing/undecodable/silent are listed under `skipped` (not scored).
    A clip that Whisper transcribes to an empty string IS scored (all deletions).
    """
    manifest_path = Path(manifest_path)
    entries = load_manifest(manifest_path)

    clips: dict[str, AudioClip] = {}
    usable: list[dict] = []
    skipped: list[dict] = []
    for e in entries:
        try:
            clips[e["id"]] = load_audio_file(manifest_path.parent / e["audio"])
            usable.append(e)
        except AudioValidationError as exc:
            skipped.append({"id": e["id"], "audio": e["audio"], "reason": str(exc)})

    rows: list[dict] = []
    failed: list[dict] = []
    total = len(models) * len(usable)
    done = 0
    for model in models:
        if usable:
            load_model(model)  # raises TranscriptionError early if unavailable
        for e in usable:
            if progress:
                progress(done, total, f"{model}: {e['id']}")
            try:
                res = transcribe_audio(clips[e["id"]].samples, model, e["language"] or None)
                rows.append(
                    {
                        "id": e["id"],
                        "model": model,
                        "language": e["language"],
                        "reference": e["reference"],
                        "hypothesis": res.text,
                        "seconds": res.seconds,
                    }
                )
            except TranscriptionError as exc:
                failed.append({"id": e["id"], "model": model, "reason": str(exc)})
            done += 1
    if progress:
        progress(total, total, "done")

    for r in rows:
        rates = compute_error_rates([r["reference"]], [r["hypothesis"]])
        r["wer"], r["cer"] = rates.wer, rates.cer

    summary: list[dict] = []
    languages = sorted({r["language"] for r in rows})
    for model in models:
        for lang in ["all", *languages]:
            s = _summarise(rows, model, lang)
            if s:
                summary.append(s)

    meta = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "manifest": str(manifest_path),
        "models": list(models),
        "device": get_device(),
        "openai_whisper_version": _whisper_version(),
        "clips_in_manifest": len(entries),
        "clips_evaluated": len(usable),
    }
    return EvaluationReport(meta=meta, summary=summary, per_clip=rows, skipped=skipped, failed=failed)


def save_report(
    report: EvaluationReport,
    json_path: Path = config.RESULTS_JSON,
    csv_path: Path = config.RESULTS_CSV,
) -> None:
    json_path, csv_path = Path(json_path), Path(csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    cols = ["id", "model", "language", "wer", "cer", "seconds", "reference", "hypothesis"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in report.per_clip:
            w.writerow({k: r[k] for k in cols})


def load_saved_report(json_path: Path = config.RESULTS_JSON) -> dict | None:
    json_path = Path(json_path)
    if not json_path.is_file():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
