"""
Прави ЧИСТ, самостоен export на верифицираниот датасет:
  - CSV со: file_name, video, start, end, duration, whisper_text (оригинал),
    corrected_text (твојата верзија), was_edited
  - копии на само тие .wav клипови што поминале review (kept/edited)

Наменето да се копира/push-не како ЗАСЕБЕН, лесен репо/папка - не целиот
vezilka проект (без код, без downloads/, без _temp_v2_new/ кеш).

Употреба:
    python export_final_dataset.py
    python export_final_dataset.py --out my_dataset_export
"""
import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset_v2_new"
PROGRESS_FILE = DATA / "review_progress.csv"
FULL_META = DATA / "metadata_full.csv"


def load_progress():
    if not PROGRESS_FILE.exists():
        raise SystemExit(f"{PROGRESS_FILE} не постои - прво пушти review_dataset.py")
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        return {r["file_name"]: r for r in csv.DictReader(f)}


def load_timestamps():
    if not FULL_META.exists():
        raise SystemExit(f"{FULL_META} не постои - прво пушти build_stt_dataset_v2_new.py")
    with open(FULL_META, encoding="utf-8") as f:
        return {r["file_name"]: r for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_export",
                     help="папка каде да се стави чистиот export (default: dataset_export/)")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    clips_out = out_dir / "clips"
    clips_out.mkdir(parents=True, exist_ok=True)

    progress = load_progress()
    timestamps = load_timestamps()

    rows = []
    missing_ts = 0
    missing_wav = 0
    for fname, r in progress.items():
        if r["status"] not in ("kept", "edited"):
            continue  # rejected клипови не влегуваат во финалниот export
        ts = timestamps.get(fname)
        if ts is None:
            missing_ts += 1
            continue

        src_wav = DATA / fname
        if not src_wav.exists():
            missing_wav += 1
            continue
        dst_wav = clips_out / Path(fname).name
        shutil.copy2(src_wav, dst_wav)

        rows.append({
            "file_name": f"clips/{dst_wav.name}",
            "video": ts.get("video", ""),
            "start": ts.get("start", ""),
            "end": ts.get("end", ""),
            "duration": ts.get("duration", ""),
            "whisper_text": r["original_text"],
            "corrected_text": r["text"],
            "was_edited": "yes" if r["status"] == "edited" else "no",
        })

    rows.sort(key=lambda r: (r["video"], float(r["start"] or 0)))

    out_csv = out_dir / "metadata.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "file_name", "video", "start", "end", "duration",
            "whisper_text", "corrected_text", "was_edited",
        ])
        w.writeheader()
        w.writerows(rows)

    edited = sum(1 for r in rows if r["was_edited"] == "yes")
    print(f"Извезени {len(rows)} клипови -> {out_dir}/")
    print(f"  {out_csv}")
    print(f"  {clips_out}/  ({len(rows)} .wav фајлови)")
    print(f"  од тие, {edited} се поправени (edited), {len(rows) - edited} биле веќе точни (kept)")
    if missing_ts:
        print(f"  ПРЕДУПРЕДУВАЊЕ: {missing_ts} редови немаа timestamp во metadata_full.csv (прескокнати)")
    if missing_wav:
        print(f"  ПРЕДУПРЕДУВАЊЕ: {missing_wav} .wav фајлови не беа најдени на диск (прескокнати)")


if __name__ == "__main__":
    main()