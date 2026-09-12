import argparse
import csv
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset_v2_new"
PROGRESS_FILE = DATA / "review_progress.csv"
FIELDS = ["file_name", "source", "original_text", "text", "status", "reviewed_at"]


def load_progress():
    if not PROGRESS_FILE.exists():
        return {}
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        return {r["file_name"]: r for r in csv.DictReader(f)}


def save_progress(progress):
    DATA.mkdir(exist_ok=True)
    with open(PROGRESS_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in progress.values():
            w.writerow(r)


def load_source_rows(source):
    fname = "metadata.csv" if source == "gold" else "metadata_silver.csv"
    p = DATA / fname
    if not p.exists():
        raise SystemExit(f"{p} not found")
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def play(path):
    try:
        subprocess.run(["afplay", str(path)], check=False)
    except FileNotFoundError:
        print("  (afplay не е достапен, отвори рачно:", path, ")")


def mark(progress, row, source, text, status):
    progress[row["file_name"]] = {
        "file_name": row["file_name"],
        "source": source,
        "original_text": row["transcription"],
        "text": text,
        "status": status,
        "reviewed_at": datetime.now().isoformat(timespec="seconds"),
    }


def review(source, limit):
    rows = load_source_rows(source)
    progress = load_progress()
    todo = [r for r in rows if r["file_name"] not in progress]
    print(f"{source}: {len(rows)} вкупно, {len(rows) - len(todo)} веќе прегледани, {len(todo)} остануваат")
    if limit:
        todo = todo[:limit]

    done = 0
    try:
        for r in todo:
            path = DATA / r["file_name"]
            text = r["transcription"]
            while True:
                print(f"\n{'=' * 60}\n{r['file_name']}")
                print(f"  текст: {text}")
                play(path)
                ans = input("  [Enter=OK, текст=коригирај, r=пушти пак, x=отфрли, q=излез] > ").strip()
                if ans == "":
                    mark(progress, r, source, text, "kept")
                    done += 1
                    break
                if ans.lower() == "r":
                    continue
                if ans.lower() == "x":
                    mark(progress, r, source, text, "rejected")
                    done += 1
                    break
                if ans.lower() == "q":
                    raise KeyboardInterrupt
                text = ans
                mark(progress, r, source, text, "edited")
                done += 1
                break
    except KeyboardInterrupt:
        print("\nпрекинато - прогресот е зачуван")
    finally:
        save_progress(progress)
        print(f"\nпрегледани во оваа сесија: {done}")
        print(f"вкупно прегледани досега: {len(progress)}")


def stats():
    progress = load_progress()
    c = Counter(r["status"] for r in progress.values())
    print("прегледани вкупно:", len(progress))
    for k, v in c.items():
        print(f"  {k}: {v}")


def export():
    progress = load_progress()
    split_map = {}
    for fname in ("metadata.csv", "metadata_silver.csv"):
        p = DATA / fname
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    split_map[r["file_name"]] = r["split"]

    rows = [
        {"file_name": r["file_name"], "transcription": r["text"],
         "split": split_map.get(r["file_name"], "train")}
        for r in progress.values() if r["status"] in ("kept", "edited")
    ]
    out_path = DATA / "metadata_verified.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "transcription", "split"])
        w.writeheader()
        w.writerows(rows)
    print(f"извезени {len(rows)} редови -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="gold", choices=["gold", "silver"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()

    if args.stats:
        stats()
    elif args.export:
        export()
    else:
        review(args.source, args.n or None)


if __name__ == "__main__":
    main()
