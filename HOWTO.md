# How to check the dataset and run models

All commands are run from the project root with the project venv:
`/Users/ristomanchev/PycharmProjects/vezilka/.venv/bin/python`

---

## 1. Look at what the build produced

```bash
.venv/bin/python inspect_dataset.py
```
Prints a table: how many GOLD / SILVER / rejected clips, total minutes,
average length, train/test counts, and the reject-reason breakdown.

Show random example rows (Whisper label vs the OCR subtitle it agreed with):
```bash
.venv/bin/python inspect_dataset.py --source gold   --n 15
.venv/bin/python inspect_dataset.py --source silver --n 15
.venv/bin/python inspect_dataset.py --grep индукц           # rows about a topic
```

**Listen** to clips and check the transcript really matches the audio
(macOS `afplay`):
```bash
.venv/bin/python inspect_dataset.py --play 8                # 8 random GOLD clips
.venv/bin/python inspect_dataset.py --source silver --play 8
```

Raw files if you prefer a spreadsheet:
| file | what |
|---|---|
| `dataset_v2_new/metadata.csv` | GOLD — `file_name,transcription,split` |
| `dataset_v2_new/metadata_silver.csv` | SILVER |
| `dataset_v2_new/metadata_full.csv` | every segment + scores + reject reason |
| `dataset_v2_new/report.txt` | the summary the build printed |
| `dataset_v2_new/clips/*.wav` | the audio clips |

---

## 2. Sanity-check a Whisper model on a real file

Transcribe any video/clip with the base model:
```bash
.venv/bin/python transcribe.py downloads/DVl1RifDEvf.mp4
.venv/bin/python transcribe.py downloads/DVl1RifDEvf.mp4 --model medium
```

Compare a dataset clip's audio against its stored label:
```bash
.venv/bin/python transcribe.py dataset_v2_new/clips/dvl1rifdevf_000003.wav --label
```

---

## 3. Fine-tune Whisper on the GOLD set

Install the training deps (one time, ~2–3 GB with torch):
```bash
.venv/bin/pip install "transformers>=4.44" "datasets>=2.20" accelerate \
    evaluate jiwer torch soundfile librosa tensorboard
```

Train (defaults to `openai/whisper-small`, GOLD only, 1000 steps):
```bash
.venv/bin/python train_whisper.py
```
Options via env vars:
```bash
BASE_MODEL=openai/whisper-medium .venv/bin/python train_whisper.py   # bigger base
INCLUDE_SILVER=1 .venv/bin/python train_whisper.py                    # add silver data
```
Output goes to `whisper-mk-finetuned/`. Training on CPU is slow — a few
hours for `small`. A GPU (or Google Colab) is strongly recommended for
`medium`/`large`.

Watch training curves:
```bash
.venv/bin/python -m tensorboard.main --logdir whisper-mk-finetuned/runs
```

---

## 4. Use the fine-tuned model

```bash
.venv/bin/python transcribe.py somefile.mp4 --hf whisper-mk-finetuned
```
The script prints the evaluation WER (word error rate) on the test split at
the end of training; lower is better. Compare it against the base
`whisper-small` WER printed at step 0.

---

## 5. If GOLD came out too small / too noisy

The Whisper transcripts and OCR are cached in `_temp_v2_new/<id>/`, so
re-classifying is fast (no re-transcription). Edit the thresholds at the top
of `build_stt_dataset_v2_new.py`:

```python
AGREEMENT_THRESHOLD = 58     # lower  -> more GOLD, more noise
WORD_RECALL_MIN     = 0.55   # lower  -> more GOLD, more noise
OCR_TIME_MARGIN     = 1.0
```
then re-run `.venv/bin/python build_stt_dataset_v2_new.py` (it reuses the caches
and only redoes the scoring + clip cutting).

To add more clean data, mix in Google **FLEURS** `mk_mk` and Mozilla
**Common Voice** `mk` (both already have verified transcripts).
