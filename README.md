# vezilka — датасет на македонски говор за Whisper

Проектот прави **датасет за speech-to-text на македонски** од кратки видеа
(Instagram Reels / TikTok) што имаат **вградени титли во сликата**. Целта е
парови `(аудио исечок, транскрипт)` за fine-tune на Whisper модел.

---

## Проблемот и решението

Видеата имаат титли „испечени" во сликата (не се посебен фајл). Наивниот
пристап — да се извади текстот со OCR и тој да се земе како транскрипт — не
работи доволно добро:

* OCR на стилизирана кирилица е многу шумен (`БЕЗ ИЗГОРЕНИЦИ` → `BE3 ИЗGOREНИЦИ`);
* времето на титлот не се совпаѓа точно со говорот (грешка ±1 сек);
* многу исечоци се по 1 секунда, една половина се само ГОЛЕМИ БУКВИ.

Затоа има **две верзии** на пайплајнот:

### v1 (стара) — `build_stt_dataset_v1_old.py`
OCR текстот директно се користи како label. Излезот е во `dataset_v1_old/`
(491 исечок). **Не се тренира на ова** — стои само за споредба.

### v2 (нова) — `build_stt_dataset_v2_new.py`  ← ова е актуелниот пристап

```
видео ──► ffmpeg ──► аудио 16 kHz mono
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
  faster-whisper (large-v3)       PaddleOCR (кириличен, mobile)
  транскрибира со VAD             чита вградени титли, 1 фрејм/сек
  + временски печати                     │
        │                               │
        ▼                               ▼
   кандидат-транскрипт            текст на екранот по секунда
   (со мали букви + интерпункција)       │
        └──────────────┬────────────────┘
                       ▼
       се споредуваат за истиот временски прозорец
       token_sort_ratio ≥ 58  И  word-recall ≥ 0.55
                       │
              ┌────────┴────────┐
              ▼                 ▼
            GOLD              SILVER
       (се согласуваат)   (само Whisper, без потврда)
```

**Клучно:** транскриптот **секогаш доаѓа од Whisper** (правилни мали букви и
интерпункција). OCR-от служи само да потврди дека зборовите се точни и дека
аудиото е порамнето со нив.

* **GOLD** = Whisper и OCR се согласуваат → голема сигурност дека транскриптот
  е точен И порамнет. **Ова се користи за тренирање.**
* **SILVER** = нема OCR потврда. Употребливо, но со помала сигурност / по рачна проверка.
* Отфрлени = прекратки/предолги, слаба доверба на Whisper, не-македонски, реклами.

---

## Што содржи репото

| Патека | Што е |
|---|---|
| `build_stt_dataset_v2_new.py` | **новиот пайплајн** (v2) |
| `build_stt_dataset_v1_old.py` | стариот пайплајн (v1), само за референца |
| `dataset_v2_new/` | **финалниот датасет** — види долу |
| `dataset_v1_old/` | стар излез (`rejected.csv` + порано `clips/`) за споредба |
| `train_whisper.py` | fine-tune на HuggingFace Whisper врз GOLD |
| `inspect_dataset.py` | статистика + слушање на исечоци (`--play`) |
| `transcribe.py` | пушти модел (base или натрениран) врз фајл |
| `download_reel.py` | симнува Reel преку `yt-dlp` |
| `tests/test_pipeline.py` | unit-тестови за чистите функции (16) |
| `requirements.txt` | зависности |
| `README_dataset.md` | подетален опис на v2 пайплајнот (EN) |
| `HOWTO.md` | чекор-по-чекор: проверка + тренирање (EN) |

**Не е во git** (по `.gitignore`): `.venv/`, кешовите `_temp_*`, `.idea/`, и
сировите видеа `downloads/` + `baba_ruza/` + `test_downloads/` (~550 MB —
влезни податоци, не резултат; се симнуваат наново со `download_reel.py`).

---

## Датасетот (`dataset_v2_new/`)

| Фајл | Содржина |
|---|---|
| `clips/*.wav` | аудио исечоци, 16 kHz mono |
| `metadata.csv` | **GOLD** — `file_name,transcription,split` (формат HuggingFace `audiofolder`) |
| `metadata_silver.csv` | SILVER — исти колони |
| `metadata_full.csv` | секој сегмент + дијагностика (оцени, `reject_reason`) |
| `report.txt` | збирни бројки од последниот прогон |

Бројки од последниот прогон (`large-v3`):

| | Исечоци | Аудио | Train / Test |
|---|---|---|---|
| GOLD | 255 | ~19 мин | 226 / 29 |
| SILVER | 409 | ~23 мин | — |
| Отфрлени | 73 | — | — |

`split` (`train`/`test`) е доделен **по видео**, за да не паднат исечоци од
истото видео и во train и во test.

---

## Брз почеток

```bash
# 1. околина
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. види што има во датасетот
.venv/bin/python inspect_dataset.py
.venv/bin/python inspect_dataset.py --source gold --n 15     # примери: Whisper vs OCR
.venv/bin/python inspect_dataset.py --play 8                 # слушни 8 GOLD исечоци (macOS afplay)

# 3. тестови
.venv/bin/python tests/test_pipeline.py

# 4. (по избор) регенерирај го датасетот — треба и видеата во downloads/ + baba_ruza/
.venv/bin/python build_stt_dataset_v2_new.py                 # симнува large-v3 (~3 GB), трае часови на CPU
WHISPER_MODEL=small MAX_VIDEOS=2 .venv/bin/python build_stt_dataset_v2_new.py   # брз тест

# 5. fine-tune
.venv/bin/pip install "transformers>=4.44" "datasets>=2.20" accelerate evaluate jiwer torch soundfile librosa tensorboard
.venv/bin/python train_whisper.py                            # GOLD, база openai/whisper-small
INCLUDE_SILVER=1 .venv/bin/python train_whisper.py           # + SILVER
```

Повеќе детали во `HOWTO.md`.

---

## Ограничувања / на што да се внимава

* **Обемот е мал** — ~19 мин GOLD. За вистински fine-tune, домешај чисти
  готови корпуси: Google **FLEURS** `mk_mk` (~10 ч) и Mozilla **Common Voice** `mk`.
  GOLD-от овде е добар како domain-adaptation додаток, не како самостоен сет.
* Дел од GOLD има ситни Whisper грешки во правописот (пр. „кујна" → „куина")
  таму каде OCR всушност бил поточен — бидејќи label-от секогаш е Whisper текстот.
* Понекогаш дијалектна каша се провлекува во GOLD (видеата од `baba_ruza`).
  Може да се исфрлат преку `VIDEO_DIRS` во скриптата, или со построг
  `WORD_RECALL_MIN`.
* Праговите (`AGREEMENT_THRESHOLD`, `WORD_RECALL_MIN`, `OCR_TIME_MARGIN`) се на
  врвот на `build_stt_dataset_v2_new.py`. Whisper+OCR резултатите се кеширани
  во `_temp_v2_new/`, па повторно прегрупирање (без ре-транскрипција) е брзо.
