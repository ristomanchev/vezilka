from pathlib import Path
import subprocess
import json
import csv
import shutil
import re
import hashlib
import os
import sys

from rapidfuzz import fuzz


ROOT_DIR = Path(__file__).resolve().parent

VIDEO_DIRS = [
    ROOT_DIR / "downloads",
    ROOT_DIR / "baba_ruza",
]

OUT_DIR = ROOT_DIR / "dataset_v2_new"
CLIPS_DIR = OUT_DIR / "clips"
CACHE_DIR = ROOT_DIR / "_temp_v2_new"

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
LANGUAGE = "mk"

INITIAL_PROMPT = "Ова е транскрипт на македонски јазик со правилна интерпункција."

OCR_FPS = 1.0
OCR_CONFIDENCE = 0.50

AGREEMENT_THRESHOLD = 58
WORD_RECALL_MIN = 0.55
OCR_TIME_MARGIN = 1.0

MIN_SEGMENT_SECONDS = 1.0
MAX_SEGMENT_SECONDS = 30.0
CLIP_PAD_SECONDS = 0.15

MAX_NO_SPEECH_PROB = 0.6
MIN_AVG_LOGPROB = -1.1
MAX_COMPRESSION_RATIO = 2.4

MIN_WORDS = 2
MIN_CYRILLIC_CHARS = 3
MAX_LATIN_RATIO = 0.5

TEST_EVERY = 10

MAX_VIDEOS_TO_PROCESS = None
if os.environ.get("MAX_VIDEOS"):
    MAX_VIDEOS_TO_PROCESS = int(os.environ["MAX_VIDEOS"])

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}

BLACKLIST_SUBSTRINGS = [
    "MILD HOME", "MILD HOME STORE", "SMART", "COLLECTION", "COLLACFIMN",
    "ANSER", "CRISTAL", "STORE", "WWW.", "HTTP", ".COM", ".MK",
    "FOLLOW", "SUBSCRIBE", "LIKE", "SHARE", "TIKTOK", "INSTAGRAM",
]


def run_command(command):
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        print("Command failed:", " ".join(str(c) for c in command))
        print(result.stderr)
        raise RuntimeError("ffmpeg command failed")


def ffprobe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


MK_LETTER_CLASS = r"А-Шабвгдѓежзѕијклљмнњопрстќуфхцчџш" + \
    "АБВГДЃЕЖЗЅИЈКЛЉМНЊОПРСТЌУФХЦЧЏШ"

_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF←-⇿⌀-⏿]",
    flags=re.UNICODE,
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_HASHTAG_RE = re.compile(r"[#@]\w+")


def clean_filename(name):
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name or "video"


def normalize_text(text):
    """Normalisation applied to the FINAL label (Whisper output)."""
    if not text:
        return ""
    text = _URL_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(" ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = text.replace("|", " ").replace("\n", " ")
    text = text.replace("“", '"').replace("”", '"').replace("„", '"')
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" -–—•*_")
    return text


def normalize_for_compare(text):
    """Aggressive normalisation used only for Whisper<->OCR similarity."""
    text = text.lower()
    text = re.sub(r"[^" + MK_LETTER_CLASS.lower() + r"a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_recall(label, ocr_text):
    """Share of Whisper content words (>=3 chars) that appear in the OCR text,
    matched fuzzily per word. Catches 'the right words are on screen' even when
    word order / spelling differ a bit; unlike token_set_ratio it is not fooled
    by a large OCR blob that merely happens to contain a few of the words."""
    label_words = [w for w in normalize_for_compare(label).split() if len(w) >= 3]
    if not label_words:
        return 0.0
    ocr_words = set(normalize_for_compare(ocr_text).split())
    if not ocr_words:
        return 0.0
    hits = 0
    for w in label_words:
        if any(fuzz.ratio(w, o) >= 80 for o in ocr_words):
            hits += 1
    return hits / len(label_words)


def agreement(label, ocr_text):
    """Return (fuzzy_score, recall). GOLD requires both to clear their thresholds."""
    if not ocr_text:
        return 0, 0.0
    a = normalize_for_compare(label)
    b = normalize_for_compare(ocr_text)
    fuzzy = int(fuzz.token_sort_ratio(a, b))
    return fuzzy, word_recall(label, ocr_text)


def count_cyrillic(text):
    return len(re.findall(r"[" + MK_LETTER_CLASS + r"]", text))


def count_latin(text):
    return len(re.findall(r"[A-Za-z]", text))


def label_is_sane(text):
    if not text:
        return False, "empty"
    words = text.split()
    if len(words) < MIN_WORDS:
        return False, "too few words"
    cyr = count_cyrillic(text)
    lat = count_latin(text)
    if cyr < MIN_CYRILLIC_CHARS:
        return False, "not enough Cyrillic"
    if cyr + lat > 0 and lat / (cyr + lat) > MAX_LATIN_RATIO:
        return False, "mostly Latin"
    upper = text.upper()
    for bad in BLACKLIST_SUBSTRINGS:
        if bad in upper:
            return False, f"blacklist:{bad}"
    low = [w.lower() for w in words]
    if len(low) >= 6 and len(set(low)) <= max(2, len(low) // 4):
        return False, "repetitive"
    return True, "ok"


def extract_audio(video_path, audio_path):
    run_command([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", str(audio_path),
    ])


def extract_frames(video_path, frames_dir):
    frames_dir.mkdir(parents=True, exist_ok=True)
    run_command([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={OCR_FPS}", str(frames_dir / "frame_%06d.png"),
    ])


def cut_audio_segment(audio_path, output_path, start, end, total_duration):
    start = max(0.0, start - CLIP_PAD_SECONDS)
    end = min(total_duration, end + CLIP_PAD_SECONDS) if total_duration else end + CLIP_PAD_SECONDS
    run_command([
        "ffmpeg", "-y", "-i", str(audio_path),
        "-ss", f"{start:.3f}", "-t", f"{max(0.05, end - start):.3f}",
        "-ac", "1", "-ar", "16000", str(output_path),
    ])


_WHISPER_MODEL_OBJ = None


def get_whisper_model():
    global _WHISPER_MODEL_OBJ
    if _WHISPER_MODEL_OBJ is None:
        from faster_whisper import WhisperModel
        print(f"Loading Whisper model '{WHISPER_MODEL}' ({WHISPER_COMPUTE_TYPE}) ...")
        _WHISPER_MODEL_OBJ = WhisperModel(
            WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
        )
    return _WHISPER_MODEL_OBJ


def transcribe_audio(audio_path, cache_path):
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    model = get_whisper_model()
    segments, info = model.transcribe(
        str(audio_path),
        language=LANGUAGE,
        task="transcribe",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        word_timestamps=True,
        condition_on_previous_text=False,
        initial_prompt=INITIAL_PROMPT,
    )

    out = []
    for s in segments:
        out.append({
            "start": s.start,
            "end": s.end,
            "text": (s.text or "").strip(),
            "avg_logprob": s.avg_logprob,
            "no_speech_prob": s.no_speech_prob,
            "compression_ratio": s.compression_ratio,
        })

    cache_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  whisper: {len(out)} raw segments  (lang_prob={info.language_probability:.2f})")
    return out


def segment_is_confident(seg):
    if seg["no_speech_prob"] > MAX_NO_SPEECH_PROB and seg["avg_logprob"] < -0.5:
        return False
    if seg["avg_logprob"] < MIN_AVG_LOGPROB:
        return False
    if seg["compression_ratio"] > MAX_COMPRESSION_RATIO:
        return False
    return True


_OCR_OBJ = None


def get_ocr():
    global _OCR_OBJ
    if _OCR_OBJ is None:
        from paddleocr import PaddleOCR
        print("Loading PaddleOCR (Cyrillic, mobile) ...")
        common = dict(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="cyrillic_PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        try:
            _OCR_OBJ = PaddleOCR(**common)
        except Exception as e:
            print("  mobile OCR init failed, falling back to lang=cyrillic:", e)
            _OCR_OBJ = PaddleOCR(lang="cyrillic")
    return _OCR_OBJ


def _paddle_texts(result):
    texts = []
    for res in result:
        data = getattr(res, "json", None)
        if callable(data):
            data = data()
        if not isinstance(data, dict) and isinstance(res, dict):
            data = res
        if not isinstance(data, dict):
            continue
        objs = [data]
        if isinstance(data.get("res"), dict):
            objs.append(data["res"])
        for obj in objs:
            for t, c in zip(obj.get("rec_texts", []), obj.get("rec_scores", [])):
                if c >= OCR_CONFIDENCE:
                    t = normalize_text(t)
                    if t:
                        texts.append(t)
    return texts


def ocr_video(video_path, frames_dir, cache_path):
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    extract_frames(video_path, frames_dir)
    ocr = get_ocr()

    results = []
    for frame_path in sorted(frames_dir.glob("frame_*.png")):
        frame_number = int(frame_path.stem.split("_")[-1])
        t = (frame_number - 1) / OCR_FPS
        try:
            output = ocr.predict(str(frame_path))
        except Exception as e:
            print("  OCR failed on", frame_path.name, e)
            continue
        text = normalize_text(" ".join(_paddle_texts(output)))
        if text:
            results.append({"time": t, "text": text})

    shutil.rmtree(frames_dir, ignore_errors=True)
    cache_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  ocr: {len(results)} frames with text")
    return results


def ocr_text_for_window(ocr_results, start, end):
    parts = []
    for item in ocr_results:
        if start - OCR_TIME_MARGIN <= item["time"] <= end + OCR_TIME_MARGIN:
            parts.append(item["text"])
    dedup = []
    for p in parts:
        if not dedup or fuzz.ratio(dedup[-1].lower(), p.lower()) < 90:
            dedup.append(p)
    return " ".join(dedup)


def process_video(video_path):
    vid = clean_filename(video_path.stem)
    cache = CACHE_DIR / vid
    cache.mkdir(parents=True, exist_ok=True)

    audio_path = cache / "audio.wav"
    if not audio_path.exists():
        extract_audio(video_path, audio_path)
    total_dur = ffprobe_duration(audio_path)

    whisper_segments = transcribe_audio(
        audio_path, cache / f"whisper-{clean_filename(WHISPER_MODEL)}.json"
    )
    ocr_results = ocr_video(video_path, cache / "frames", cache / "ocr-mobile.json")

    rows = []
    for i, seg in enumerate(whisper_segments):
        start, end = seg["start"], seg["end"]
        dur = end - start
        label = normalize_text(seg["text"])

        base = {
            "video": video_path.name,
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(dur, 2),
            "text": label,
            "avg_logprob": round(seg["avg_logprob"], 3),
            "no_speech_prob": round(seg["no_speech_prob"], 3),
            "ocr_text": "",
            "ocr_score": 0,
            "ocr_recall": 0.0,
            "source": "",
            "reject_reason": "",
        }

        if dur < MIN_SEGMENT_SECONDS:
            base["source"] = "rejected"; base["reject_reason"] = "too short"
            rows.append(base); continue
        if dur > MAX_SEGMENT_SECONDS:
            base["source"] = "rejected"; base["reject_reason"] = "too long"
            rows.append(base); continue
        if not segment_is_confident(seg):
            base["source"] = "rejected"; base["reject_reason"] = "low whisper confidence"
            rows.append(base); continue

        sane, why = label_is_sane(label)
        if not sane:
            base["source"] = "rejected"; base["reject_reason"] = why
            rows.append(base); continue

        ocr_text = ocr_text_for_window(ocr_results, start, end)
        score, recall = agreement(label, ocr_text)
        base["ocr_text"] = ocr_text
        base["ocr_score"] = score
        base["ocr_recall"] = round(recall, 2)
        is_gold = score >= AGREEMENT_THRESHOLD and recall >= WORD_RECALL_MIN
        base["source"] = "gold" if is_gold else "silver"

        clip_name = f"{vid}_{i:06d}.wav"
        cut_audio_segment(audio_path, CLIPS_DIR / clip_name, start, end, total_dur)
        base["file_name"] = f"clips/{clip_name}"
        rows.append(base)

    n_gold = sum(1 for r in rows if r["source"] == "gold")
    n_silver = sum(1 for r in rows if r["source"] == "silver")
    print(f"  -> gold={n_gold}  silver={n_silver}  rejected={len(rows) - n_gold - n_silver}")
    return rows


def split_for_video(video_name):
    h = int(hashlib.md5(video_name.encode()).hexdigest(), 16)
    return "test" if h % TEST_EVERY == 0 else "train"


def collect_videos():
    videos = []
    for d in VIDEO_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(p)
    return videos


def main():
    OUT_DIR.mkdir(exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    videos = collect_videos()
    if MAX_VIDEOS_TO_PROCESS is not None:
        videos = videos[:MAX_VIDEOS_TO_PROCESS]
    print(f"Found {len(videos)} videos")
    if not videos:
        return

    all_rows = []
    for idx, video_path in enumerate(videos, 1):
        print(f"\n[{idx}/{len(videos)}] {video_path.name}")
        try:
            all_rows.extend(process_video(video_path))
        except KeyboardInterrupt:
            print("Interrupted by user - writing what we have so far.")
            break
        except Exception as e:
            print("  FAILED:", e)
            all_rows.append({
                "video": video_path.name, "source": "rejected",
                "reject_reason": f"processing failed: {e}",
                "start": "", "end": "", "duration": "", "text": "",
                "avg_logprob": "", "no_speech_prob": "", "ocr_text": "",
                "ocr_score": "", "file_name": "",
            })

    for r in all_rows:
        r["split"] = split_for_video(r.get("video", "")) if r.get("source") in ("gold", "silver") else ""

    full_cols = ["file_name", "text", "source", "split", "video", "start", "end",
                 "duration", "ocr_score", "ocr_recall", "ocr_text", "avg_logprob",
                 "no_speech_prob", "reject_reason"]
    with open(OUT_DIR / "metadata_full.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=full_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    def write_subset(path, source):
        rows = [r for r in all_rows if r.get("source") == source]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file_name", "transcription", "split"])
            w.writeheader()
            for r in rows:
                w.writerow({"file_name": r["file_name"],
                            "transcription": r["text"],
                            "split": r["split"]})
        return len(rows)

    n_gold = write_subset(OUT_DIR / "metadata.csv", "gold")
    n_silver = write_subset(OUT_DIR / "metadata_silver.csv", "silver")

    gold = [r for r in all_rows if r.get("source") == "gold"]
    silver = [r for r in all_rows if r.get("source") == "silver"]
    rejected = [r for r in all_rows if r.get("source") == "rejected"]

    def total_minutes(rows):
        return sum(float(r["duration"]) for r in rows if r.get("duration")) / 60

    report = [
        "Macedonian STT dataset - build report",
        "=" * 44,
        f"videos processed              : {len(videos)}",
        f"GOLD  clips (whisper+OCR agree): {n_gold}   ~{total_minutes(gold):.1f} min",
        f"SILVER clips (whisper only)    : {n_silver}   ~{total_minutes(silver):.1f} min",
        f"rejected segments             : {len(rejected)}",
        "",
        f"gold  train/test : {sum(1 for r in gold if r['split']=='train')} / {sum(1 for r in gold if r['split']=='test')}",
        f"whisper model    : {WHISPER_MODEL}",
        f"agreement thresh : {AGREEMENT_THRESHOLD}",
        "",
        "Reject reasons:",
    ]
    from collections import Counter
    for reason, n in Counter(r.get("reject_reason", "") for r in rejected).most_common():
        report.append(f"  {n:5d}  {reason}")
    report_text = "\n".join(report)
    (OUT_DIR / "report.txt").write_text(report_text + "\n", encoding="utf-8")

    print("\n" + report_text)
    print(f"\nWritten to {OUT_DIR}/")
    print("  metadata.csv         (GOLD - use this to fine-tune Whisper)")
    print("  metadata_silver.csv  (extra, lower confidence)")
    print("  metadata_full.csv    (all segments + diagnostics)")


if __name__ == "__main__":
    sys.exit(main())
