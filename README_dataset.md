# Macedonian STT dataset pipeline

Goal: turn short Macedonian videos with burned-in subtitles into
`(audio clip, transcript)` pairs for fine-tuning Whisper.

## Why not just OCR the subtitles?

The first approach (`build_stt_dataset_v1_old.py`) used PaddleOCR of the on-screen
subtitles directly as labels. That output is not trainable:

* OCR of stylised Cyrillic is very noisy (`БЕЗ ИЗГОРЕНИЦИ` -> `BE3 ИЗGOREНИЦИ`);
* subtitle timing (sampled at 1 fps) is ±1 s off from the audio;
* ~78 % of clips were ≤1 s single words, ~52 % were ALL-CAPS.

## The v2 pipeline (`build_stt_dataset_v2_new.py`)

```
video ──► ffmpeg ──► 16 kHz mono wav
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
 faster-whisper                    PaddleOCR (mobile, Cyrillic)
 VAD segmentation                  reads burned-in subtitles
 word timestamps                   1 frame/sec
        │                               │
        ▼                               ▼
 candidate label                  on-screen text per second
 (cased + punctuated)                    │
        └──────────────┬────────────────┘
                       ▼
        agreement(label, ocr_text_in_window)
        fuzzy score  >= 55   AND
        word recall  >= 0.55
                       │
              ┌────────┴────────┐
              ▼                 ▼
            GOLD              SILVER
     metadata.csv        metadata_silver.csv
```

* **Label text always comes from Whisper** (properly cased/punctuated).
  OCR is only used to *confirm* the words and the alignment.
* **GOLD** = Whisper and OCR agree -> high confidence the transcript is
  correct *and* aligned with the audio. Use this to fine-tune.
* **SILVER** = no OCR confirmation. Usable with lower confidence / after a
  manual pass.
* Segments outside 1–30 s, low Whisper confidence, non-Macedonian, or
  brand/spam text are rejected (see `reject_reason` in `metadata_full.csv`).

### Outputs (`dataset_v2_new/`)

| file | contents |
|---|---|
| `clips/*.wav` | 16 kHz mono audio clips |
| `metadata.csv` | GOLD — `file_name,transcription,split` (HF `audiofolder`) |
| `metadata_silver.csv` | SILVER — same columns |
| `metadata_full.csv` | every segment + diagnostics (scores, logprobs, reject reason) |
| `report.txt` | summary counts |

`split` is `train`/`test`, assigned per source video so clips from one video
never straddle the split.

## Running

```bash
.venv/bin/pip install -r requirements.txt

# full run, best quality (downloads whisper large-v3 ~3 GB, slow on CPU)
.venv/bin/python build_stt_dataset_v2_new.py

# quick check on 2 videos with a small model
WHISPER_MODEL=small MAX_VIDEOS=2 .venv/bin/python build_stt_dataset_v2_new.py
```

Per-video results are cached in `_temp_v2_new/<id>/` (`audio.wav`,
`whisper-<model>.json`, `ocr-mobile.json`) so re-runs only redo what changed.
Delete a cache file to force recompute.

### Tuning knobs (top of the script)

| constant | meaning |
|---|---|
| `WHISPER_MODEL` | `large-v3` (best) / `medium` / `small` |
| `AGREEMENT_THRESHOLD`, `WORD_RECALL_MIN` | how strict GOLD is |
| `OCR_FPS` | frames/sec sent to OCR |
| `MIN/MAX_SEGMENT_SECONDS` | clip length limits |
| `VIDEO_DIRS` | which folders to read (drop `baba_ruza` if the heavy dialect hurts) |
| `TEST_EVERY` | 1-in-N videos to the test split |

## Fine-tuning

```bash
.venv/bin/pip install "transformers>=4.44" "datasets>=2.20" accelerate \
    evaluate jiwer torch soundfile librosa tensorboard
.venv/bin/python train_whisper.py          # trains on GOLD
INCLUDE_SILVER=1 .venv/bin/python train_whisper.py   # + silver
```

## Tests

```bash
.venv/bin/python tests/test_pipeline.py     # or: python -m pytest tests/ -q
```

## Also worth adding (clean, already-labelled Macedonian speech)

* Google **FLEURS** `mk_mk` (~10 h read speech)
* Mozilla **Common Voice** `mk`

Mixing a few hours of those with the GOLD clips here gives a much stronger
fine-tune than this scraped data alone (~tens of minutes).
