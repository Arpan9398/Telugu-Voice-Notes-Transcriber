# Telugu Voice Notes Transcriber

Upload or record a voice note in **Telugu** or **Hindi**, transcribe it with **OpenAI Whisper**
(Base and Small side by side), pick the transcript you prefer, and measure accuracy with
**Word Error Rate** (Jiwer).

Stack: Python, openai-whisper, Streamlit, librosa, jiwer.

## Features

- Upload (wav, mp3, m4a, flac, ogg, webm) or record from the microphone
- Telugu, Hindi, or auto-detect
- Whisper **Base** vs **Small** shown side by side, with inference time
- Choose a preferred transcript, edit it, download as `.txt`
- Optional: paste a reference transcript to get WER/CER for the current clip
- Evaluation set (`data/eval/manifest.csv`) + `evaluate.py` CLI + an in-app results tab (WER, CER, S/D/I)
- Handles empty uploads, empty recordings, corrupt files, silence, too-short and too-long audio

## Project layout

```
app.py                       Streamlit UI
evaluate.py                  CLI: WER evaluation of base/small on the manifest
requirements.txt / requirements-dev.txt / .env.example
tvt/
  config.py                  settings (env vars)
  audio.py                   librosa loading + validation
  transcriber.py             Whisper model cache + transcribe
  metrics.py                 text normalisation + WER/CER (jiwer)
  evaluation.py              manifest runner, summaries, save/load results
scripts/generate_sample_audio.py   make sample clips from the manifest sentences (gTTS)
data/eval/manifest.csv       10 reference sentences (5 Telugu, 5 Hindi)
data/eval/audio/             clips go here (not bundled, see below)
results/                     evaluation output is written here
tests/                       pytest tests
```

## Setup

Requires Python 3.10+ and, for non-wav formats (m4a, webm, ...), **ffmpeg** on your PATH
(`brew install ffmpeg` / `sudo apt install ffmpeg` / `winget install ffmpeg`).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # optional
```

Whisper weights are **not** in this project; they download automatically on first use
(Base is about 140 MB, Small about 460 MB) into `~/.cache/whisper` (or `WHISPER_MODEL_DIR`).

## Run the app

```bash
streamlit run app.py
```

## Evaluate accuracy (WER)

The manifest has correct reference transcripts but **no audio is bundled**. Generate the clips
(needs internet; uses Google TTS, so they are synthetic speech):

```bash
python scripts/generate_sample_audio.py
python evaluate.py                 # base + small
python evaluate.py --models base   # one model
```

Results print as a table and are saved to `results/wer_results.json` and `results/wer_per_clip.csv`;
the **Accuracy evaluation** tab in the app shows them too.

### Use real recordings (recommended)

TTS speech is much cleaner than real voice notes, so its WER says little about real-world use.
Record your own clips, save them in `data/eval/audio/`, and add rows to `data/eval/manifest.csv`
(`id,audio,language,reference`; `language` is `te` or `hi`; `audio` is relative to the manifest),
or point at another file with `python evaluate.py --manifest path/to/manifest.csv`.

### How scoring works

- Reference and hypothesis are normalised before scoring: Unicode NFC, case-folded, zero-width
  characters removed, punctuation (including the Devanagari danda) removed.
- WER = (substitutions + deletions + insertions) / reference words, summed over clips (corpus level),
  using `jiwer.process_words`. CER is computed the same way on characters.
- A clip Whisper transcribes to nothing is scored as all deletions. Clips whose audio is missing,
  undecodable or silent are skipped and listed, not scored.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Configuration (`.env`)

`WHISPER_MODEL_DIR`, `WHISPER_DEVICE` (cpu/cuda/mps), `MAX_AUDIO_SECONDS` (600),
`MIN_AUDIO_SECONDS` (0.5), `SILENCE_RMS_THRESHOLD` (0.001).

## Notes and limitations

- Whisper's Telugu support on the Base and Small models is limited compared with larger models;
  expect noticeably worse results than for English. Use the evaluation tab to measure it on your own audio.
- CPU inference works but is slow for long clips; Small is slower than Base.
- The whole clip is processed at once (limit: `MAX_AUDIO_SECONDS`).
