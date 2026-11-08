"""Telugu Voice Notes Transcriber - Streamlit app.   Run: streamlit run app.py"""
from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from tvt import config
from tvt.audio import AudioValidationError, load_audio_bytes
from tvt.evaluation import evaluate_models, load_manifest, load_saved_report, save_report
from tvt.metrics import compute_error_rates
from tvt.transcriber import TranscriptionError, get_device, transcribe_audio

st.set_page_config(page_title="Telugu Voice Notes Transcriber", page_icon="🎙️", layout="wide")

UPLOAD_TYPES = ["wav", "mp3", "m4a", "flac", "ogg", "webm"]


# ---------------------------------------------------------------- transcribe tab
def _get_audio_input():
    options = ["Upload a file"]
    if hasattr(st, "audio_input"):
        options.append("Record from microphone")
    mode = st.radio("Audio source", options, horizontal=True)
    if mode == "Upload a file":
        f = st.file_uploader("Upload a voice note", type=UPLOAD_TYPES)
    else:
        f = st.audio_input("Record a voice note")
    if f is None:
        return None, None
    return f.getvalue(), getattr(f, "name", None) or "recording.wav"


def render_transcribe_tab(language_code: str | None, models: list[str]) -> None:
    data, filename = _get_audio_input()
    if data is None:
        st.info("Upload or record audio to begin.")
        return
    if len(data) == 0:
        st.error("The upload/recording is empty (0 bytes). Please try again.")
        return

    st.audio(data)
    key = hashlib.sha1(data).hexdigest()[:12] + f"-{language_code}-{'+'.join(models)}"

    if st.button("Transcribe", type="primary", disabled=not models):
        try:
            clip = load_audio_bytes(data, filename)
        except AudioValidationError as exc:
            st.session_state.pop("transcription", None)
            st.error(str(exc))
            return
        results, errors = {}, {}
        for m in models:
            with st.spinner(f"Running Whisper {m} on {get_device()}... (first run downloads the model)"):
                try:
                    results[m] = transcribe_audio(clip.samples, m, language_code)
                except TranscriptionError as exc:
                    errors[m] = str(exc)
        st.session_state["transcription"] = {
            "key": key, "results": results, "errors": errors, "duration": clip.duration,
        }

    state = st.session_state.get("transcription")
    if not state or state["key"] != key:
        return

    st.caption(f"Audio length: {state['duration']:.1f}s")
    for m, msg in state["errors"].items():
        st.error(f"Whisper {m}: {msg}")
    results = state["results"]
    if not results:
        return

    st.subheader("Model comparison")
    for col, (m, res) in zip(st.columns(len(results)), results.items()):
        with col:
            st.markdown(f"**Whisper {m}**")
            st.caption(f"{res.seconds:.1f}s inference | language: {res.language}")
            if res.is_empty:
                st.warning("No speech detected by this model.")
            st.text_area("Transcript", res.text, height=200, disabled=True, key=f"ro-{m}-{key}")

    st.subheader("Your preferred transcript")
    preferred = st.radio("Preferred model", list(results), horizontal=True, key=f"pref-{key}")
    final = st.text_area("Edit if needed", results[preferred].text, height=160, key=f"final-{key}-{preferred}")
    st.download_button("Download .txt", final.encode("utf-8"), file_name="transcript.txt",
                       mime="text/plain; charset=utf-8", disabled=not final.strip())

    with st.expander("Score against a reference transcript (optional)"):
        ref = st.text_area("Correct transcript", key=f"ref-{key}")
        if ref.strip():
            rows = []
            for m, res in results.items():
                try:
                    r = compute_error_rates([ref], [res.text])
                    rows.append({"Model": m, "WER %": round(r.wer * 100, 1), "CER %": round(r.cer * 100, 1),
                                 "Sub": r.substitutions, "Del": r.deletions, "Ins": r.insertions})
                except ValueError as exc:
                    st.warning(str(exc))
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True)


# ---------------------------------------------------------------- evaluation tab
def _summary_frame(report: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "Model": s["model"], "Language": s["language"], "Clips": s["clips"], "Ref words": s["ref_words"],
            "WER %": round(s["wer"] * 100, 1), "CER %": round(s["cer"] * 100, 1),
            "Sub": s["substitutions"], "Del": s["deletions"], "Ins": s["insertions"],
            "Avg s/clip": round(s["avg_seconds"], 2),
        } for s in report["summary"]]
    )


def render_eval_tab() -> None:
    try:
        entries = load_manifest()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    base = config.EVAL_MANIFEST.parent
    table = pd.DataFrame(
        [{"id": e["id"], "language": e["language"], "reference": e["reference"],
          "audio present": (base / e["audio"]).is_file()} for e in entries]
    )
    st.markdown("**Evaluation set** (`data/eval/manifest.csv`)")
    st.dataframe(table, hide_index=True)
    missing = int((~table["audio present"]).sum())
    if missing:
        st.warning(
            f"{missing} of {len(table)} audio clips are missing. Generate them with "
            "`python scripts/generate_sample_audio.py` (needs internet), or add your own "
            "recordings and edit the manifest."
        )

    models = st.multiselect("Models to evaluate", list(config.MODEL_CHOICES), default=list(config.MODEL_CHOICES))
    if st.button("Run evaluation", type="primary", disabled=(not models or missing == len(table))):
        bar = st.progress(0.0, text="Starting...")

        def progress(done: int, total: int, msg: str) -> None:
            bar.progress(done / total if total else 1.0, text=msg)

        try:
            report = evaluate_models(models, progress=progress)
        except (TranscriptionError, ValueError, FileNotFoundError) as exc:
            bar.empty()
            st.error(str(exc))
        else:
            bar.empty()
            if report.per_clip:
                save_report(report)
                st.success("Evaluation finished and saved to results/.")
            else:
                st.error("No clips could be evaluated. Check the audio files listed in the manifest.")

    saved = load_saved_report()
    if not saved:
        st.info("No evaluation results yet. Run the evaluation above or `python evaluate.py`.")
        return

    meta = saved["meta"]
    st.markdown(f"**Latest results** - {meta['created_utc']} UTC, device `{meta['device']}`, "
                f"openai-whisper {meta['openai_whisper_version']}, "
                f"{meta['clips_evaluated']}/{meta['clips_in_manifest']} clips evaluated")
    summary = _summary_frame(saved)
    st.dataframe(summary, hide_index=True)
    overall = summary[summary["Language"] == "all"]
    if not overall.empty:
        st.bar_chart(overall.set_index("Model")[["WER %", "CER %"]])
    for s in saved.get("skipped", []):
        st.warning(f"Skipped {s['id']}: {s['reason']}")
    for f in saved.get("failed", []):
        st.warning(f"Failed {f['id']} ({f['model']}): {f['reason']}")
    with st.expander("Per-clip results"):
        st.dataframe(pd.DataFrame(saved["per_clip"]), hide_index=True)
    st.caption("Lower is better. WER can exceed 100%. Sample clips are synthetic TTS speech unless "
               "you replaced them, so treat these numbers as a smoke test, not real-world accuracy.")


# ---------------------------------------------------------------- layout
def main() -> None:
    st.title("🎙️ Telugu Voice Notes Transcriber")
    st.caption("Whisper Base vs Small - Telugu and Hindi speech to text")

    with st.sidebar:
        st.header("Settings")
        lang_label = st.selectbox("Spoken language", list(config.LANGUAGES))
        models = st.multiselect("Models to compare", list(config.MODEL_CHOICES), default=list(config.MODEL_CHOICES))
        st.caption(f"Device: `{get_device()}`  \nMax audio: {config.MAX_AUDIO_SECONDS / 60:.0f} min")

    tab1, tab2 = st.tabs(["Transcribe", "Accuracy evaluation"])
    with tab1:
        render_transcribe_tab(config.LANGUAGES[lang_label], models)
    with tab2:
        render_eval_tab()


main()
