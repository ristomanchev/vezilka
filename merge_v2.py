"""
Го спојува dataset_merged (del1+del2+del3) со dataset_v2_new од main
(clips + metadata_popravka.csv) во еден финален датасет.

metadata_popravka.csv е Excel верзија од metadata_full.csv (со ; и со
поправен текст). Затоа:
  - whisper_text  = text од metadata_full.csv (оригиналот)
  - corrected_text = text од metadata_popravka.csv (твојата поправка)
  - start/end/duration/video се земаат од metadata_full.csv (Excel ги расипува бројките)
  - rejected редови (без клип) се прескокнуваат
  - видеа каде ниеден ред не е поправен се сметаат за непрегледани и се
    прескокнуваат (исклучи со --include-unreviewed)

Употреба:
    python merge_v2_popravka.py --v2-dir ..\\vezilka_main\\dataset_v2_new
"""
import argparse
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIELDS = ["file_name", "video", "start", "end", "duration",
          "whisper_text", "corrected_text", "was_edited"]


def clean(t):
    return re.sub(r"\s+", " ", (t or "")).strip()


def read_csv_any(path):
    """Чита CSV без разлика дали е со , или ; и дали има ѓубре пред header-от."""
    raw = None
    for enc in ("utf-8-sig", "cp1251"):
        try:
            raw = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if raw is None:
        raise SystemExit(f"Не можам да го прочитам {path}")
    lines = raw.splitlines(keepends=True)
    start = next((i for i, l in enumerate(lines) if l.lstrip().startswith("file_name")), None)
    if start is None:
        raise SystemExit(f"{path}: нема ред со колони (file_name...)")
    header = lines[start]
    delim = ";" if header.count(";") > header.count(",") else ","
    return list(csv.DictReader(lines[start:], delimiter=delim))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default="dataset_merged")
    ap.add_argument("--v2-dir", required=True, help="dataset_v2_new од main branch")
    ap.add_argument("--out", default="dataset_final")
    ap.add_argument("--include-unreviewed", action="store_true")
    args = ap.parse_args()

    merged_dir = (ROOT / args.merged).resolve()
    v2_dir = Path(args.v2_dir).resolve()
    out_dir = (ROOT / args.out).resolve()
    clips_out = out_dir / "clips"
    clips_out.mkdir(parents=True, exist_ok=True)

    rows, seen = [], set()

    # 1) dataset_merged - веќе во финален формат
    for r in read_csv_any(merged_dir / "metadata.csv"):
        name = Path(r["file_name"]).name
        src = merged_dir / r["file_name"]
        if not src.exists():
            print(f"  нема {src}, прескокнато")
            continue
        shutil.copy2(src, clips_out / name)
        seen.add(name)
        rows.append({k: r[k] for k in FIELDS})
    print(f"{args.merged}: {len(rows)} клипови")

    # 2) dataset_v2_new од main
    full = {r["file_name"]: r for r in read_csv_any(v2_dir / "metadata_full.csv") if r["file_name"]}
    pop = [r for r in read_csv_any(v2_dir / "metadata_popravka.csv") if r.get("file_name")]

    v2_rows = []
    for p in pop:
        f = full.get(p["file_name"])
        if f is None:
            print(f"  {p['file_name']} го нема во metadata_full.csv, прескокнато")
            continue
        if p.get("source") == "rejected":
            continue
        orig, corr = clean(f["text"]), clean(p["text"])
        if not corr:
            continue
        v2_rows.append({
            "file_name": p["file_name"], "video": f["video"],
            "start": f["start"], "end": f["end"], "duration": f["duration"],
            "whisper_text": orig, "corrected_text": corr,
            "was_edited": "yes" if orig != corr else "no",
        })

    edits_per_video = defaultdict(int)
    for r in v2_rows:
        edits_per_video[r["video"]] += r["was_edited"] == "yes"
    unreviewed = sorted(v for v in edits_per_video if edits_per_video[v] == 0)
    if unreviewed and not args.include_unreviewed:
        print(f"  Прескокнати {len(unreviewed)} непрегледани видеа (0 поправки): {', '.join(unreviewed)}")
        v2_rows = [r for r in v2_rows if r["video"] not in unreviewed]

    added = 0
    for r in v2_rows:
        src = v2_dir / r["file_name"]
        name = Path(r["file_name"]).name
        if not src.exists():
            print(f"  нема {src}, прескокнато")
            continue
        if name in seen:
            print(f"  ДУПЛИКАТ {name}, прескокнато")
            continue
        shutil.copy2(src, clips_out / name)
        seen.add(name)
        r["file_name"] = f"clips/{name}"
        rows.append(r)
        added += 1
    print(f"dataset_v2_new (popravka): {added} клипови")

    rows.sort(key=lambda r: (r["video"], float(r["start"] or 0)))
    with open(out_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    minutes = sum(float(r["duration"] or 0) for r in rows) / 60
    edited = sum(r["was_edited"] == "yes" for r in rows)
    print(f"\nВкупно {len(rows)} клипови од {len({r['video'] for r in rows})} видеа "
          f"({minutes:.1f} мин), {edited} поправени -> {out_dir}")


if __name__ == "__main__":
    main()