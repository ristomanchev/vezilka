"""
Bulk-download source videos into downloads/ for the STT dataset pipeline.

Usage
-----
    # one or more URLs on the command line
    python download_reel.py https://www.instagram.com/reel/XXXX/ https://vm.tiktok.com/YYYY/

    # a text file with one URL per line ('#' comments and blank lines ignored)
    python download_reel.py urls.txt

    # a whole channel / playlist / profile — yt-dlp expands it to every video
    python download_reel.py https://www.tiktok.com/@someaccount
    python download_reel.py "https://www.youtube.com/@somechannel/shorts" --max 200

    # interactive (old behaviour) if no arguments
    python download_reel.py

Notes
-----
* Already-downloaded videos are skipped (recorded in downloads/.download-archive).
* Errors on one URL do not stop the rest; failures are listed at the end and
  appended to downloads/failed.txt.
* --max N limits how many NEW videos to pull this run (useful for channels).
"""

import sys
from pathlib import Path

import yt_dlp

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "downloads"
ARCHIVE = OUT_DIR / ".download-archive"
FAILED_LOG = OUT_DIR / "failed.txt"


def collect_urls(args):
    urls = []
    for a in args:
        p = Path(a)
        if p.exists() and p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.split("#", 1)[0].strip()
                if line:
                    urls.append(line)
        else:
            urls.append(a)
    seen = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


def main(argv):
    max_new = None
    if "--max" in argv:
        i = argv.index("--max")
        max_new = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]

    if not argv:
        argv = [input("Enter video/reel/channel URL: ").strip()]

    urls = collect_urls(argv)
    if not urls:
        print("no URLs given")
        return 1

    OUT_DIR.mkdir(exist_ok=True)
    print(f"{len(urls)} URL(s) to process; new videos land in {OUT_DIR}/")

    before = {p.name for p in OUT_DIR.glob("*.mp4")}
    failed = []

    ydl_opts = {
        "outtmpl": str(OUT_DIR / "%(id)s.%(ext)s"),
        "format": "mp4/bestvideo*+bestaudio/best",
        "download_archive": str(ARCHIVE),
        "ignoreerrors": True,
        "quiet": False,
        "noprogress": False,
        "retries": 3,
        "concurrent_fragment_downloads": 4,
    }
    if max_new is not None:
        ydl_opts["max_downloads"] = max_new

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in urls:
            try:
                ydl.download([url])
            except yt_dlp.utils.MaxDownloadsReached:
                print(f"reached --max {max_new}, stopping")
                break
            except Exception as e:
                print(f"FAILED: {url}\n   {e}")
                failed.append(url)

    after = {p.name for p in OUT_DIR.glob("*.mp4")}
    new = sorted(after - before)
    print(f"\nDONE. {len(new)} new video(s) this run, {len(after)} total in downloads/")
    if failed:
        FAILED_LOG.parent.mkdir(exist_ok=True)
        with open(FAILED_LOG, "a", encoding="utf-8") as f:
            for u in failed:
                f.write(u + "\n")
        print(f"{len(failed)} URL(s) failed - see {FAILED_LOG}")

    print("\nNext: python build_stt_dataset_v2_new.py   (only the new videos get transcribed)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
