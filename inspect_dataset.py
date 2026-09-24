import argparse
import csv
import random
import shutil
import subprocess
import sys
import wave
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CANDIDATES = [
    (ROOT / "dataset" / "metadata.csv", "corrected_text", "whisper_text", "was_edited"),
    (ROOT / "dataset_raw" / "metadata_full.csv", "text", "ocr_text", "source"),
]


def detect_columns(fields):
    text_col = next((c for c in ("corrected_text", "transcription", "text") if c in fields), "text")
    orig_col = next((c for c in ("whisper_text", "ocr_text") if c in fields), "")
    group_col = next((c for c in ("was_edited", "source") if c in fields), "")
    return text_col, orig_col, group_col


def read_rows(explicit=None):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"{p} not found")
        with open(p, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return (rows, p.parent) + detect_columns(rows[0].keys() if rows else [])

    for p, text_col, orig_col, group_col in CANDIDATES:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return list(csv.DictReader(f)), p.parent, text_col, orig_col, group_col

    raise SystemExit("no dataset found (looked for dataset/ and dataset_raw/)")


def wav_seconds(path):
    try:
        with wave.open(str(path)) as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


def minutes(rows):
    return sum(float(r["duration"]) for r in rows if r.get("duration")) / 60


def summarise(rows, base, text_col, group_col):
    groups = {}
    for r in rows:
        groups.setdefault(r.get(group_col, "-"), []).append(r)

    print("=" * 62)
    print(f"{base.name}" + (f"   (групирано по {group_col})" if group_col else ""))
    print("-" * 62)
    print(f"{'група':14} {'клипови':>8} {'минути':>9} {'просек с':>9}")
    for g in sorted(groups):
        rs = groups[g]
        secs = [float(r["duration"]) for r in rs if r.get("duration")]
        avg = (sum(secs) / len(secs)) if secs else 0
        print(f"{str(g):14} {len(rs):>8} {sum(secs)/60:>9.1f} {avg:>9.2f}")
    print("-" * 62)
    all_secs = [float(r["duration"]) for r in rows if r.get("duration")]
    print(f"{'ВКУПНО':14} {len(rows):>8} {sum(all_secs)/60:>9.1f} "
          f"{sum(all_secs)/len(all_secs) if all_secs else 0:>9.2f}")
    print("=" * 62)

    vids = {r["video"] for r in rows if r.get("video")}
    if vids:
        print(f"изворни видеа: {len(vids)}")

    if rows and "dialect" in rows[0]:
        dia = [r for r in rows if r["dialect"] == "yes"]
        dv = {r["video"] for r in dia}
        print(f"дијалект:      {len(dia)} клипа од {len(dv)} видеа ({minutes(dia):.1f} мин)")
        print(f"стандарден:    {len(rows)-len(dia)} клипа ({minutes(rows)-minutes(dia):.1f} мин)")

    wl = [len(r[text_col].split()) for r in rows if r.get(text_col)]
    if wl:
        print(f"зборови/клип:  мин {min(wl)}  медијана {sorted(wl)[len(wl)//2]}  макс {max(wl)}")

    missing = sum(1 for r in rows if r.get("file_name") and not (base / r["file_name"]).exists())
    if missing:
        print(f"ВНИМАНИЕ: {missing} реда покажуваат кон непостоечки .wav")

    if rows and "reject_reason" in rows[0]:
        reasons = Counter(r["reject_reason"] for r in rows if r.get("reject_reason"))
        if reasons:
            print("\nпричини за отфрлање:")
            for reason, n in reasons.most_common():
                print(f"  {n:5d}  {reason}")


def filter_rows(rows, group_col, group, grep, text_col, dialect):
    rs = rows
    if group and group_col:
        rs = [r for r in rs if r.get(group_col) == group]
    if dialect:
        rs = [r for r in rs if r.get("dialect") == dialect]
    if grep:
        rs = [r for r in rs if grep.lower() in r.get(text_col, "").lower()]
    return rs


def show_samples(rows, text_col, orig_col, group_col, group, n, grep, dialect):
    rs = filter_rows(rows, group_col, group, grep, text_col, dialect)
    random.shuffle(rs)
    tags = [t for t in (f"{group_col}={group}" if group else "",
                        f"dialect={dialect}" if dialect else "",
                        f"текст~{grep!r}" if grep else "") if t]
    print(f"\n--- {min(n, len(rs))} од {len(rs)} примери "
          f"[{', '.join(tags) if tags else 'сите'}] ---")
    for r in rs[:n]:
        extra = "  ".join(f"{c}={r[c]}" for c in (group_col, "dialect") if c and c in r)
        print(f"\n{r['file_name']}   [{r.get('duration', '?')}s]  {extra}")
        print(f"  ТЕКСТ    : {r.get(text_col, '')}")
        orig = r.get(orig_col, "")
        if orig and orig != r.get(text_col):
            print(f"  ОРИГИНАЛ : {orig[:140]}")


def play_command(path):
    if sys.platform == "darwin":
        return ["afplay", str(path)]
    if sys.platform == "win32":
        return ["cmd", "/c", "start", "/wait", "", str(path)]
    for player in ("aplay", "paplay", "ffplay"):
        if shutil.which(player):
            if player == "ffplay":
                return [player, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]
            return [player, str(path)]
    return ["xdg-open", str(path)]


def play(rows, base, text_col, group_col, group, n, grep, dialect):
    rs = [r for r in filter_rows(rows, group_col, group, grep, text_col, dialect)
          if r.get("file_name")]
    random.shuffle(rs)
    for r in rs[:n]:
        path = base / r["file_name"]
        if not path.exists():
            continue
        print(f"\n> {path.name}  [{r.get('duration', '?')}s]")
        print(f"  {r.get(text_col, '')}")
        cmd = play_command(path)
        try:
            subprocess.run(cmd, check=False)
        except (FileNotFoundError, OSError):
            print(f"  (не можам да пуштам аудио со '{cmd[0]}' - отвори рачно: {path})")
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="", help="патека до metadata csv (default: dataset/)")
    ap.add_argument("--group", default="", help="филтрирај по was_edited (yes/no) или source (gold/silver)")
    ap.add_argument("--dialect", default="", choices=["", "yes", "no"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--grep", default="")
    ap.add_argument("--play", type=int, default=0, metavar="N")
    args = ap.parse_args()

    rows, base, text_col, orig_col, group_col = read_rows(args.csv)
    summarise(rows, base, text_col, group_col)
    if args.play:
        play(rows, base, text_col, group_col, args.group, args.play, args.grep, args.dialect)
    else:
        show_samples(rows, text_col, orig_col, group_col, args.group, args.n, args.grep, args.dialect)


if __name__ == "__main__":
    main()
