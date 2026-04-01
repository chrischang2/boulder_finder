"""
Full scraper pipeline: search → filter → download → extract frames.

This is the main entry-point for the scraping workflow.  It chains
together the Instagram scraper, boulder-name parser, and frame
extractor into one convenient CLI.

Usage examples
──────────────
    # Credentials are loaded automatically from .env (IG_USERNAME / IG_PASSWORD)
    python -m scraper.pipeline

    # Or override via CLI flags
    python -m scraper.pipeline --username my_ig_user --password my_ig_pass

    # URL-list mode — provide specific post URLs (no login needed)
    python -m scraper.pipeline --urls https://instagram.com/reel/ABC123 https://instagram.com/p/DEF456

    # URL-list from a file (one URL per line)
    python -m scraper.pipeline --url-file urls.txt

    # Use browser cookies for yt-dlp downloads (avoids login issues)
    python -m scraper.pipeline --url-file urls.txt --cookies-from chrome

    # Scan more posts and extract frames every 0.5 s
    python -m scraper.pipeline --max-posts 500 --frame-interval 0.5

    # Only extract frames from already-downloaded videos
    python -m scraper.pipeline --frames-only
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from scraper.instagram_scraper import InstagramScraper, DEFAULT_HASHTAG, SCRAPE_DIR
from scraper.browser_scraper import scrape_hashtag_browser
from scraper.frame_extractor import extract_all, extract_frames
from scraper.boulder_parser import KNOWN_BOULDERS

# ── Load .env if present ─────────────────────────────────────────────
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"


def _load_dotenv() -> None:
    """Read key=value pairs from .env into os.environ (if the file exists)."""
    if not _ENV_PATH.is_file():
        return
    with open(_ENV_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key:
                os.environ.setdefault(key, value)


_load_dotenv()


def run_pipeline(
    hashtag: str = DEFAULT_HASHTAG,
    max_posts: int = 200,
    max_videos: int = 5,
    username: str | None = None,
    password: str | None = None,
    output_dir: Path = SCRAPE_DIR,
    frame_interval: float = 1.0,
    frames_only: bool = False,
    urls: list[str] | None = None,
    cookies_from: str | None = None,
    use_api: bool = False,
    headless: bool = False,
) -> None:
    """
    Execute the full scrape-and-extract pipeline.

    Steps
    -----
    1. Print known boulder names so the user can verify.
    2. Search #hashtag, filter by caption, download matching videos.
    3. Extract frames from every downloaded video.
    4. Write a manifest JSON summarising what was scraped.
    """
    print("=" * 60)
    print("  Boulder Finder — Instagram Scraper Pipeline")
    print("=" * 60)
    print(f"\n  Hashtag     : #{hashtag}")
    print(f"  Max videos  : {max_videos}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Frame every : {frame_interval}s")
    print(f"  Known boulders ({len(KNOWN_BOULDERS)}):")
    for name, folder in sorted(KNOWN_BOULDERS.items()):
        print(f"    • {name.title():30s}  →  {folder}/")
    print()

    # ── Step 1: Scrape ───────────────────────────────────────────────
    results: list[dict] = []

    if not frames_only:
        if urls:
            # URL-list mode — use API scraper with direct URLs
            scraper = InstagramScraper(
                username=username,
                password=password,
                output_dir=output_dir,
                cookies_from=cookies_from,
            )
            results = scraper.scrape_urls(urls)
        elif use_api:
            # API mode — use instaloader (may fail due to IG restrictions)
            scraper = InstagramScraper(
                username=username,
                password=password,
                output_dir=output_dir,
                cookies_from=cookies_from,
            )
            results = scraper.scrape_hashtag(hashtag=hashtag, max_posts=max_posts)
        else:
            # Default: browser mode — most reliable
            results = scrape_hashtag_browser(
                hashtag=hashtag,
                username=username,
                password=password,
                max_videos=max_videos,
                output_dir=output_dir,
                headless=headless,
            )

    # ── Step 2: Extract frames ───────────────────────────────────────
    print("\n[pipeline] Extracting frames from downloaded videos …\n")
    n_processed = extract_all(
        scrape_root=output_dir,
        every_n_seconds=frame_interval,
    )
    print(f"\n[pipeline] Frame extraction complete — {n_processed} videos processed.")

    # ── Step 3: Write manifest ───────────────────────────────────────
    if results:
        manifest_path = output_dir / "manifest.json"

        # Append to existing manifest if present
        existing: list[dict] = []
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError):
                existing = []

        for r in results:
            r["scraped_at"] = datetime.now().isoformat()

        existing.extend(results)
        manifest_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n[pipeline] Manifest updated: {manifest_path}")

    print("\n[pipeline] All done ✓\n")


# ── CLI ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Instagram for Castle Hill bouldering videos, "
                    "filter by boulder name, download, and extract frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hashtag", default=DEFAULT_HASHTAG,
        help="Instagram hashtag to search (default: %(default)s)",
    )
    parser.add_argument(
        "--max-posts", type=int, default=200,
        help="Max posts to scan in API mode (default: 200)",
    )
    parser.add_argument(
        "--max-videos", type=int, default=5,
        help="Max matching videos to download (default: 5)",
    )
    parser.add_argument(
        "--username", default=None,
        help="Instagram username for authenticated access (recommended)",
    )
    parser.add_argument(
        "--password", default=None,
        help="Instagram password",
    )
    parser.add_argument(
        "--output-dir", default=str(SCRAPE_DIR),
        help="Root output directory (default: data/scraped/)",
    )
    parser.add_argument(
        "--frame-interval", type=float, default=1.0,
        help="Seconds between extracted frames (default: 1.0)",
    )
    parser.add_argument(
        "--frames-only", action="store_true",
        help="Skip scraping, only extract frames from existing videos",
    )
    parser.add_argument(
        "--urls", nargs="*",
        help="Instagram post/reel URLs to download directly (no login needed)",
    )
    parser.add_argument(
        "--url-file",
        help="Path to a text file with one Instagram URL per line",
    )
    parser.add_argument(
        "--cookies-from", default=None,
        help="Browser to extract cookies from for yt-dlp (chrome/firefox/edge)",
    )
    parser.add_argument(
        "--use-api", action="store_true",
        help="Use instaloader API instead of browser (may be blocked by IG)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run browser without GUI window",
    )

    args = parser.parse_args()

    # CLI flags override .env values
    username = args.username or os.environ.get("IG_USERNAME")
    password = args.password or os.environ.get("IG_PASSWORD")
    cookies_from = args.cookies_from or os.environ.get("IG_COOKIES_FROM")

    # Collect URLs
    url_list: list[str] | None = None
    if args.urls:
        url_list = args.urls
    elif args.url_file:
        url_list = Path(args.url_file).read_text(encoding="utf-8").splitlines()

    run_pipeline(
        hashtag=args.hashtag,
        max_posts=args.max_posts,
        max_videos=args.max_videos,
        username=username,
        password=password,
        output_dir=Path(args.output_dir),
        frame_interval=args.frame_interval,
        frames_only=args.frames_only,
        urls=url_list,
        cookies_from=cookies_from,
        use_api=args.use_api,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
