"""
Instagram scraper for Castle Hill bouldering videos.

Supports two modes:
  1. **Hashtag search** — uses `instaloader` to search #castlehillbouldering,
     filter by boulder name, then download with `yt-dlp`.
  2. **URL list** — provide a text file of Instagram post URLs (one per line)
     and the scraper will check captions for boulder names and download matches.

Downloaded assets are stored under:
    data/scraped/<boulder_folder>/<shortcode>/
        video.mp4
        caption.txt

Prerequisites
─────────────
    pip install instaloader yt-dlp
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Iterator

import instaloader  # type: ignore

from scraper.boulder_parser import BoulderMatch, find_boulders

# ── Defaults ─────────────────────────────────────────────────────────
DEFAULT_HASHTAG = "castlehillbouldering"
SCRAPE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "scraped"
REQUEST_DELAY_S = 2.0   # polite delay between API requests
_SHORTCODE_RE = re.compile(r"instagram\.com/(?:p|reel|tv)/([A-Za-z0-9_-]+)")


# ── yt-dlp helpers ───────────────────────────────────────────────────

def _ytdlp_download(url: str, dest_dir: Path, cookies_from: str | None = None) -> Path | None:
    """
    Download a video from *url* into *dest_dir*/video.mp4 using yt-dlp.

    Returns the path to the downloaded file, or None on failure.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    output_path = dest_dir / "video.mp4"

    if output_path.exists():
        return output_path

    cmd = [
        "yt-dlp",
        "--no-warnings",
        "--quiet",
        "-f", "mp4/best",
        "-o", str(output_path),
        url,
    ]
    if cookies_from:
        cmd.extend(["--cookies-from-browser", cookies_from])

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        if output_path.exists():
            return output_path
    except subprocess.CalledProcessError as exc:
        print(f"  [yt-dlp error] {exc.stderr.strip()}")
    except FileNotFoundError:
        print("  [error] yt-dlp not found — install with: pip install yt-dlp")
    except subprocess.TimeoutExpired:
        print(f"  [error] yt-dlp timed out for {url}")

    return None


def _ytdlp_get_description(url: str, cookies_from: str | None = None) -> str | None:
    """Fetch video description/caption via yt-dlp without downloading."""
    cmd = [
        "yt-dlp",
        "--no-warnings",
        "--quiet",
        "--skip-download",
        "--print", "%(description)s",
        url,
    ]
    if cookies_from:
        cmd.extend(["--cookies-from-browser", cookies_from])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


# ── Scraper class ────────────────────────────────────────────────────

class InstagramScraper:
    """
    Scraper for Instagram boulder videos.

    Parameters
    ----------
    username : str | None
        Instagram username for instaloader session (for hashtag search).
    password : str | None
        Instagram password.
    output_dir : Path
        Root folder for scraped data.  Defaults to ``data/scraped/``.
    delay : float
        Seconds to sleep between requests to avoid rate limits.
    cookies_from : str | None
        Browser name to extract cookies from for yt-dlp
        (e.g. "chrome", "firefox", "edge").  Helps with login-walled content.
    """

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        output_dir: Path = SCRAPE_DIR,
        delay: float = REQUEST_DELAY_S,
        cookies_from: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.cookies_from = cookies_from
        self._username = username
        self._password = password

        # Lazy-init instaloader only when needed for hashtag search
        self._loader: instaloader.Instaloader | None = None

    def _get_loader(self) -> instaloader.Instaloader:
        """Initialise and return an instaloader instance (lazy)."""
        if self._loader is not None:
            return self._loader

        self._loader = instaloader.Instaloader(
            download_videos=False,      # we use yt-dlp for downloads
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=False,
            compress_json=False,
            post_metadata_txt_pattern="",
        )

        if self._username and self._password:
            print(f"[scraper] Logging in as {self._username} …")
            try:
                self._loader.load_session_from_file(self._username)
                print("[scraper] Loaded saved session.")
            except FileNotFoundError:
                print("[scraper] No saved session — logging in fresh …")
                self._loader.login(self._username, self._password)
                self._loader.save_session_to_file()
                print("[scraper] Session saved for next time.")
        else:
            print("[scraper] No credentials — hashtag search may fail.")

        return self._loader

    # ── Download a single post ───────────────────────────────────────

    def _process_post(
        self,
        url: str,
        caption: str,
        shortcode: str,
    ) -> list[dict]:
        """
        Check caption for boulder names, download video if matched.

        Returns list of result dicts (one per boulder matched).
        """
        boulders = find_boulders(caption)
        if not boulders:
            return []

        results: list[dict] = []
        for b in boulders:
            dest = self.output_dir / b.folder_name / shortcode
            if dest.exists() and (dest / "video.mp4").exists():
                print(f"  [skip] already have {shortcode} ({b.label})")
                continue

            dest.mkdir(parents=True, exist_ok=True)

            # Save caption for auditing
            (dest / "caption.txt").write_text(caption, encoding="utf-8")

            # Download video via yt-dlp
            print(f"  [downloading] {shortcode} → {b.label} …")
            video_path = _ytdlp_download(url, dest, cookies_from=self.cookies_from)

            if video_path:
                print(f"  [ok] {b.label} → {dest.relative_to(self.output_dir)}")
                results.append({
                    "shortcode": shortcode,
                    "boulder": b.name,
                    "grade": b.grade,
                    "folder": str(dest),
                })
            else:
                print(f"  [fail] Could not download {shortcode}")

        return results

    # ── Mode 1: Hashtag search ───────────────────────────────────────

    def scrape_hashtag(
        self,
        hashtag: str = DEFAULT_HASHTAG,
        max_posts: int = 200,
    ) -> list[dict]:
        """
        Search #hashtag for videos mentioning known boulders.

        Uses instaloader for the search and yt-dlp for the download.
        """
        print(f"\n[scraper] Searching #{hashtag} (up to {max_posts} posts) …\n")
        loader = self._get_loader()

        results: list[dict] = []
        scanned = 0

        try:
            ht = instaloader.Hashtag.from_name(loader.context, hashtag)
            for post in ht.get_posts():
                if scanned >= max_posts:
                    break
                scanned += 1
                caption = post.caption or ""

                if not post.is_video:
                    continue

                url = f"https://www.instagram.com/p/{post.shortcode}/"
                results.extend(
                    self._process_post(url, caption, post.shortcode)
                )
                time.sleep(self.delay)

        except Exception as exc:
            print(f"\n[scraper] Hashtag search error: {exc}")
            print("[scraper] Try using --urls mode instead, or provide "
                  "--cookies-from browser_name.\n")

        print(f"\n[scraper] Scanned {scanned} posts, "
              f"downloaded {len(results)} videos.\n")
        return results

    # ── Mode 2: URL list ─────────────────────────────────────────────

    def scrape_urls(self, urls: list[str]) -> list[dict]:
        """
        Download videos from a list of Instagram URLs.

        For each URL, fetches the caption via yt-dlp, checks for boulder
        names, and downloads if matched.
        """
        print(f"\n[scraper] Processing {len(urls)} URLs …\n")
        results: list[dict] = []

        for i, url in enumerate(urls, 1):
            url = url.strip()
            if not url:
                continue

            # Extract shortcode from URL
            m = _SHORTCODE_RE.search(url)
            if not m:
                print(f"  [{i}] Cannot parse shortcode from: {url}")
                continue
            shortcode = m.group(1)

            print(f"  [{i}/{len(urls)}] {shortcode} …")

            # Get caption
            caption = _ytdlp_get_description(url, cookies_from=self.cookies_from) or ""
            if not caption:
                print(f"    [warn] No caption retrieved for {shortcode}")

            post_results = self._process_post(url, caption, shortcode)
            if not post_results and caption:
                print(f"    [skip] No known boulder names in caption")
            results.extend(post_results)

            time.sleep(self.delay)

        print(f"\n[scraper] Done. Downloaded {len(results)} videos.\n")
        return results

    # ── Unified entry-point ──────────────────────────────────────────

    def scrape(
        self,
        hashtag: str = DEFAULT_HASHTAG,
        max_posts: int = 200,
        urls: list[str] | None = None,
    ) -> list[dict]:
        """
        Run the scraper in the appropriate mode.

        If *urls* is provided, uses URL-list mode.
        Otherwise falls back to hashtag search.
        """
        if urls:
            return self.scrape_urls(urls)
        return self.scrape_hashtag(hashtag=hashtag, max_posts=max_posts)


# ── CLI convenience ──────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape Instagram for boulder videos."
    )
    parser.add_argument("--hashtag", default=DEFAULT_HASHTAG,
                        help="Hashtag to search (default: %(default)s)")
    parser.add_argument("--max-posts", type=int, default=200,
                        help="Max posts to scan (default: 200)")
    parser.add_argument("--urls", nargs="*",
                        help="Instagram post URLs to download directly")
    parser.add_argument("--url-file",
                        help="Path to a text file with one URL per line")
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--cookies-from", default=None,
                        help="Browser to extract cookies from (chrome/firefox/edge)")
    parser.add_argument("--output-dir", default=str(SCRAPE_DIR))
    args = parser.parse_args()

    url_list: list[str] | None = None
    if args.urls:
        url_list = args.urls
    elif args.url_file:
        url_list = Path(args.url_file).read_text(encoding="utf-8").splitlines()

    scraper = InstagramScraper(
        username=args.username,
        password=args.password,
        output_dir=Path(args.output_dir),
        cookies_from=args.cookies_from,
    )
    scraper.scrape(hashtag=args.hashtag, max_posts=args.max_posts, urls=url_list)
