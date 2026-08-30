"""
Inspect the dataset produced by build_stt_dataset_v2_new.py.

    .venv/bin/python inspect_dataset.py               # summary + a few samples
    .venv/bin/python inspect_dataset.py --n 20        # more samples
    .venv/bin/python inspect_dataset.py --source silver
    .venv/bin/python inspect_dataset.py --play 5      # open 5 GOLD clips in your audio player
    .venv/bin/python inspect_dataset.py --grep индукц # rows whose text matches

The --play option uses `afplay` on macOS so you can listen and check the
transcript really matches the audio.
"""

import argparse
import csv
import random
import subprocess
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset_v2_new"


def read_full():
    p = DATA / "metadata_full.csv"
    if not p.exists():
        raise SystemExit(f"{p} not found - run build_stt_dataset_v2_new.py first")
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wav_seconds(path):
    try:
        with wave.open(str(path)) as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


def summarise(rows):
    by_source = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)

    print("=" * 60)
    print(f"{'source':10} {'clips':>7} {'minutes':>9} {'avg sec':>9}")
    print("-" * 60)
    for src in ("gold", "silver", "rejected"):
        rs = by_source.get(src, [])
        secs = [float(r["duration"]) for r in rs if r.get("duration")]
        mins = sum(secs) / 60 if secs else 0
        avg = (sum(secs) / len(secs)) if secs else 0
        print(f"{src:10} {len(rs):>7} {mins:>9.1f} {avg:>9.2f}")
    print("=" * 60)

    gold = by_source.get("gold", [])
    if gold:
        vids = {r["video"] for r in gold}
        tr = sum(1 for r in gold if r["split"] == "train")
        te = sum(1 for r in gold if r["split"] == "test")
        print(f"GOLD spans {len(vids)} videos   train/test clips = {tr}/{te}")
        wl = [len(r["text"].split()) for r in gold]
        print(f"GOLD words/clip: min {min(wl)}  median {sorted(wl)[len(wl)//2]}  max {max(wl)}")

    rej = by_source.get("rejected", [])
    if rej:
        from collections import Counter
        print("\nreject reasons:")
        for reason, n in Counter(r["reject_reason"] for r in rej).most_common():
            print(f"  {n:5d}  {reason}")


def show_samples(rows, source, n, grep):
    rs = [r for r in rows if r["source"] == source]
    if grep:
        rs = [r for r in rs if grep.lower() in r["text"].lower()]
    random.shuffle(rs)
    print(f"\n--- {min(n, len(rs))} {source} samples "
          f"{'matching ' + repr(grep) if grep else ''} ---")
    for r in rs[:n]:
        print(f"\n{r['file_name']}   [{r['duration']}s]  "
              f"score={r.get('ocr_score')} recall={r.get('ocr_recall')}")
        print(f"  WHISPER: {r['text']}")
        if r.get("ocr_text"):
            print(f"  OCR    : {r['ocr_text'][:120]}")


def play(rows, source, n):
    rs = [r for r in rows if r["source"] == source and r["file_name"]]
    random.shuffle(rs)
    for r in rs[:n]:
        path = DATA / r["file_name"]
        print(f"\n▶ {path.name}  [{r['duration']}s]")
        print(f"  transcript: {r['text']}")
        try:
            subprocess.run(["afplay", str(path)], check=False)
        except FileNotFoundError:
            print("  (afplay not available - open the file manually:", path, ")")
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="gold", choices=["gold", "silver", "rejected"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--grep", default="")
    ap.add_argument("--play", type=int, default=0, metavar="N")
    args = ap.parse_args()

    rows = read_full()
    summarise(rows)
    if args.play:
        play(rows, args.source, args.play)
    else:
        show_samples(rows, args.source, args.n, args.grep)


if __name__ == "__main__":
    main()
