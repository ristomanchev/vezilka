import yt_dlp

def download_instagram_reel(url):
    ydl_opts = {
        "outtmpl": "downloads/%(id)s.%(ext)s",
        "format": "mp4",
        "quiet": False
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


if __name__ == "__main__":
    reel_url = input("Enter Instagram Reel URL: ").strip()
    download_instagram_reel(reel_url)