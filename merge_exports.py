"""
Ги спојува повеќе export папки (dataset_export, dataset_export_del2,
dataset_export_del3, ...) во една: заеднички metadata.csv + сите .wav клипови.

Употреба (од root на проектот):
    python merge_exports.py
    python merge_exports.py --inputs dataset_export dataset_export_del2 dataset_export_del3 --out dataset_merged
"""
import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIELDS = ["file_name", "video", "start", "end", "duration",
          "whisper_text", "corrected_text", "was_edited"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+",
                    default=["dataset_export", "dataset_export_del2", "dataset_export_del3"])
    ap.add_argument("--out", default="dataset_merged")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    clips_out = out_dir / "clips"
    clips_out.mkdir(parents=True, exist_ok=True)

    rows, seen = [], {}
    for name in args.inputs:
        src = ROOT / name
        meta = src / "metadata.csv"
        if not meta.exists():
            raise SystemExit(f"Не постои {meta}")
        with open(meta, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != FIELDS:
                raise SystemExit(f"{meta}: различни колони {reader.fieldnames}")
            added = 0
            for r in reader:
                wav_name = Path(r["file_name"]).name
                src_wav = src / r["file_name"]
                if not src_wav.exists():
                    print(f"  ПРЕДУПРЕДУВАЊЕ: нема {src_wav}, прескокнато")
                    continue
                if wav_name in seen:
                    print(f"  ДУПЛИКАТ: {wav_name} (веќе од {seen[wav_name]}), прескокнато")
                    continue
                seen[wav_name] = name
                shutil.copy2(src_wav, clips_out / wav_name)
                r["file_name"] = f"clips/{wav_name}"
                rows.append(r)
                added += 1
        print(f"{name}: {added} клипови")

    rows.sort(key=lambda r: (r["video"], float(r["start"] or 0)))
    with open(out_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    minutes = sum(float(r["duration"] or 0) for r in rows) / 60
    edited = sum(r["was_edited"] == "yes" for r in rows)
    videos = len({r["video"] for r in rows})
    print(f"\nВкупно {len(rows)} клипови од {videos} видеа ({minutes:.1f} мин), "
          f"{edited} поправени -> {out_dir}/")


if __name__ == "__main__":
    main()