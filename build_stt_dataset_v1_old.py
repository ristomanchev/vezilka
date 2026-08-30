from pathlib import Path
import subprocess
import csv
import shutil
import re
from rapidfuzz import fuzz
from paddleocr import PaddleOCR


ROOT_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT_DIR / "downloads"
DATASET_DIR = ROOT_DIR / "dataset_v1_old"
CLIPS_DIR = DATASET_DIR / "clips"
TEMP_DIR = ROOT_DIR / "_temp_v1_old"

# Lower FPS = faster OCR.
# FPS = 1 means OCR checks 1 frame per second.
# You can use 0.5 for 1 frame every 2 seconds.
FPS = 1

OCR_CONFIDENCE = 0.50
SIMILARITY_THRESHOLD = 85

MIN_SEGMENT_SECONDS = 0.7
MAX_SEGMENT_SECONDS = 20.0

# Text that appears too many times is probably a logo, poster, background text, etc.
STATIC_TEXT_MAX_REPETITIONS = 8

# For small testing, set this to 1, 3, 5, etc.
# For full run, set it to None.
MAX_VIDEOS_TO_PROCESS = None
# Example:
# MAX_VIDEOS_TO_PROCESS = 3


def run_command(command):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print("Command failed:")
        print(" ".join(command))
        print(result.stderr)
        raise RuntimeError("FFmpeg command failed")


def clean_filename(name):
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return name or "video"


def clean_text(text):
    text = text.strip()
    text = text.replace("|", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    # Keep Macedonian Cyrillic, Latin letters, numbers and normal punctuation.
    text = re.sub(
        r"[^А-ШЃЖЗЅИЈЉЊЌУФХЦЧЏа-шѓжзѕијљњќуфхцчџA-Za-z0-9.!?,:'\"()\-\s]+",
        "",
        text
    )

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_cyrillic_chars(text):
    return len(re.findall(r"[А-ШЃЖЗЅИЈЉЊЌУФХЦЧЏа-шѓжзѕијљњќуфхцчџ]", text))


def count_latin_chars(text):
    return len(re.findall(r"[A-Za-z]", text))


def count_digits(text):
    return len(re.findall(r"\d", text))


def is_good_transcript(text):
    text = clean_text(text)

    if not text:
        return False

    words = text.split()

    # Too short is often not useful or is logo text.
    if len(words) < 2:
        return False

    # Too long is often a poster/list/background paragraph.
    if len(words) > 18:
        return False

    cyrillic_count = count_cyrillic_chars(text)
    latin_count = count_latin_chars(text)
    digit_count = count_digits(text)
    total_letters = cyrillic_count + latin_count

    # Macedonian subtitles should contain Cyrillic.
    if cyrillic_count < 3:
        return False

    # Remove codes/numbers like 22X36, 1500M8000M, etc.
    if digit_count > 2:
        return False

    # Remove mostly English/logo text.
    if total_letters > 0 and latin_count / total_letters > 0.50:
        return False

    # Remove common background/logo/brand words.
    blacklist = [
        "NIKE",
        "LTE",
        "LECTRONIC",
        "ELECTRONIC",
        "LAB",
        "TECH",
        "22X36",
        "1500",
        "8000",
        "AURA",
        "POINTS",
        "MAIN",
        "CHARACTER",
        "INTERNATIONAL",
        "OLYMPIAD",
        "STEM",
        "RIZZ",
        "MINECRAFT",
        "CREEPER",
        "IQ UP",
    ]

    upper_text = text.upper()

    for bad in blacklist:
        if bad in upper_text:
            return False

    return True


def extract_audio(video_path, audio_path):
    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        str(audio_path)
    ]

    run_command(command)


def extract_frames(video_path, frames_dir):
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Full-frame OCR because subtitles appear in different areas.
    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={FPS}",
        str(frames_dir / "frame_%06d.png")
    ]

    run_command(command)


def extract_text_from_paddle_result(result):
    texts = []

    for res in result:
        data = None

        # PaddleOCR 3.x usually returns result objects.
        if hasattr(res, "json"):
            data = res.json

            if callable(data):
                data = data()

        elif isinstance(res, dict):
            data = res

        if not isinstance(data, dict):
            continue

        possible_data_objects = [data]

        # Some PaddleOCR versions store useful data inside "res".
        if "res" in data and isinstance(data["res"], dict):
            possible_data_objects.append(data["res"])

        for obj in possible_data_objects:
            rec_texts = obj.get("rec_texts", [])
            rec_scores = obj.get("rec_scores", [])

            for detected_text, confidence in zip(rec_texts, rec_scores):
                if confidence >= OCR_CONFIDENCE:
                    detected_text = clean_text(detected_text)

                    if detected_text:
                        texts.append(detected_text)

    return texts


def ocr_frames(frames_dir, ocr):
    results = []

    frame_paths = sorted(frames_dir.glob("frame_*.png"))

    for frame_path in frame_paths:
        frame_number = int(frame_path.stem.split("_")[-1])
        time_seconds = (frame_number - 1) / FPS

        try:
            output = ocr.predict(str(frame_path))
        except Exception as e:
            print("OCR failed on frame:", frame_path.name)
            print(e)
            continue

        texts = extract_text_from_paddle_result(output)
        full_text = clean_text(" ".join(texts))

        if full_text:
            print("OCR FOUND:", full_text)

            results.append({
                "time": time_seconds,
                "text": full_text
            })

    return results


def similar_text(a, b):
    return fuzz.ratio(a.lower(), b.lower()) >= SIMILARITY_THRESHOLD


def remove_static_text(ocr_results, max_repetitions=STATIC_TEXT_MAX_REPETITIONS):
    """
    Removes text that appears too many times.
    This helps remove background posters, logos, titles, and static screen text.
    """

    filtered = []
    rejected = []

    for item in ocr_results:
        text = item["text"]
        repetitions = 0

        for other in ocr_results:
            if similar_text(text, other["text"]):
                repetitions += 1

        if repetitions <= max_repetitions:
            filtered.append(item)
        else:
            print("REMOVED STATIC TEXT:", text)
            rejected.append({
                "time": item["time"],
                "text": text,
                "reason": "static/repeated text"
            })

    return filtered, rejected


def group_ocr_results(ocr_results):
    if not ocr_results:
        return []

    segments = []

    current_text = ocr_results[0]["text"]
    start_time = ocr_results[0]["time"]
    end_time = ocr_results[0]["time"]

    for item in ocr_results[1:]:
        t = item["time"]
        text = item["text"]

        if similar_text(current_text, text):
            end_time = t
        else:
            segments.append({
                "start": start_time,
                "end": end_time + (1 / FPS),
                "text": current_text
            })

            current_text = text
            start_time = t
            end_time = t

    segments.append({
        "start": start_time,
        "end": end_time + (1 / FPS),
        "text": current_text
    })

    return segments


def cut_audio_segment(audio_path, output_path, start, end):
    duration = end - start

    command = [
        "ffmpeg",
        "-y",
        "-i", str(audio_path),
        "-ss", str(start),
        "-t", str(duration),
        "-ac", "1",
        "-ar", "16000",
        str(output_path)
    ]

    run_command(command)


def process_video(video_path, ocr):
    print()
    print("Processing:", video_path.name)

    video_id = clean_filename(video_path.stem)

    video_temp_dir = TEMP_DIR / video_id
    frames_dir = video_temp_dir / "frames"
    audio_path = video_temp_dir / "audio.wav"

    if video_temp_dir.exists():
        shutil.rmtree(video_temp_dir)

    video_temp_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting audio...")
    extract_audio(video_path, audio_path)

    print("Extracting frames...")
    extract_frames(video_path, frames_dir)

    print("Running OCR...")
    ocr_results = ocr_frames(frames_dir, ocr)

    print("OCR detections before static filtering:", len(ocr_results))

    ocr_results, rejected_static = remove_static_text(
        ocr_results,
        max_repetitions=STATIC_TEXT_MAX_REPETITIONS
    )

    print("OCR detections after static filtering:", len(ocr_results))

    segments = group_ocr_results(ocr_results)

    print("Grouped segments:", len(segments))

    rows = []
    rejected_rows = []

    clip_index = 0

    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = clean_text(segment["text"])
        duration = end - start

        if not text:
            rejected_rows.append({
                "video": video_path.name,
                "start": start,
                "end": end,
                "text": text,
                "reason": "empty text"
            })
            continue

        if not is_good_transcript(text):
            print("REMOVED BAD TEXT:", text)

            rejected_rows.append({
                "video": video_path.name,
                "start": start,
                "end": end,
                "text": text,
                "reason": "bad transcript filter"
            })
            continue

        if duration < MIN_SEGMENT_SECONDS:
            rejected_rows.append({
                "video": video_path.name,
                "start": start,
                "end": end,
                "text": text,
                "reason": "too short audio segment"
            })
            continue

        if duration > MAX_SEGMENT_SECONDS:
            rejected_rows.append({
                "video": video_path.name,
                "start": start,
                "end": end,
                "text": text,
                "reason": "too long audio segment"
            })
            continue

        output_filename = f"{video_id}_{clip_index:06d}.wav"
        output_path = CLIPS_DIR / output_filename

        cut_audio_segment(audio_path, output_path, start, end)

        rows.append({
            "audio": f"clips/{output_filename}",
            "text": text
        })

        clip_index += 1

    for rejected in rejected_static:
        rejected_rows.append({
            "video": video_path.name,
            "start": rejected["time"],
            "end": rejected["time"],
            "text": rejected["text"],
            "reason": rejected["reason"]
        })

    print("Saved clips:", len(rows))
    print("Rejected rows:", len(rejected_rows))

    return rows, rejected_rows


def main():
    DATASET_DIR.mkdir(exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)

    video_extensions = [".mp4", ".mov", ".mkv", ".webm", ".avi"]

    videos = [
        path for path in VIDEOS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in video_extensions
    ]

    videos = sorted(videos)

    if MAX_VIDEOS_TO_PROCESS is not None:
        videos = videos[:MAX_VIDEOS_TO_PROCESS]

    print("Found videos:", len(videos))

    if not videos:
        print("No videos found in downloads/")
        return

    print("Loading OCR model...")

    # Macedonian uses Cyrillic.
    # This loads the Cyrillic OCR recognition model.
    ocr = PaddleOCR(lang="mk")

    all_rows = []
    all_rejected_rows = []

    for video_path in videos:
        try:
            rows, rejected_rows = process_video(video_path, ocr)
            all_rows.extend(rows)
            all_rejected_rows.extend(rejected_rows)
        except KeyboardInterrupt:
            print()
            print("Stopped manually by user.")
            break
        except Exception as e:
            print("Failed to process:", video_path.name)
            print(e)

            all_rejected_rows.append({
                "video": video_path.name,
                "start": "",
                "end": "",
                "text": "",
                "reason": f"processing failed: {e}"
            })

    metadata_path = DATASET_DIR / "metadata.csv"
    rejected_path = DATASET_DIR / "rejected.csv"

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "text"])
        writer.writeheader()
        writer.writerows(all_rows)

    with open(rejected_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video", "start", "end", "text", "reason"]
        )
        writer.writeheader()
        writer.writerows(all_rejected_rows)

    print()
    print("DONE")
    print("Total accepted dataset rows:", len(all_rows))
    print("Total rejected rows:", len(all_rejected_rows))
    print("Dataset folder:", DATASET_DIR)
    print("Accepted metadata:", metadata_path)
    print("Rejected metadata:", rejected_path)


if __name__ == "__main__":
    main()