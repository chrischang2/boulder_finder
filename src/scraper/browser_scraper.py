"""
Browser-based Instagram hashtag scraper using Playwright.

Uses a real Chromium browser to:
  1. Log into Instagram
  2. Navigate to the hashtag explore page
  3. Scroll through posts, collecting video URLs and captions
  4. Download matched videos via yt-dlp

This bypasses all API restrictions because it behaves like a real user.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright, Page

from scraper.boulder_parser import find_boulders

SCRAPE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "scraped"
_SESSION_DIR = Path(__file__).resolve().parent.parent.parent / ".browser_session"
_URL_LOG = SCRAPE_DIR / "downloaded_urls.txt"
_MULTI_BOULDER_LOG = SCRAPE_DIR / "multi_boulder_urls.txt"
_NO_MATCH_LOG = SCRAPE_DIR / "no_match_urls.txt"


def _load_url_log() -> set[str]:
    """Load previously-downloaded shortcodes from the URL log file."""
    if not _URL_LOG.is_file():
        return set()
    codes: set[str] = set()
    for line in _URL_LOG.read_text(encoding="utf-8").splitlines():
        token = line.strip().split()[0] if line.strip() else ""
        if token:
            codes.add(token)
    return codes


def _append_url_log(shortcode: str, post_url: str) -> None:
    """Append a shortcode + URL pair to the dedup log."""
    _URL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_URL_LOG, "a", encoding="utf-8") as f:
        f.write(f"{shortcode} {post_url}\n")


def _append_multi_boulder_log(
    shortcode: str, post_url: str, boulders: list
) -> None:
    """Log posts that reference multiple boulders for future processing."""
    _MULTI_BOULDER_LOG.parent.mkdir(parents=True, exist_ok=True)
    names = ", ".join(b.label for b in boulders)
    with open(_MULTI_BOULDER_LOG, "a", encoding="utf-8") as f:
        f.write(f"{shortcode} {post_url} | {names}\n")


def _load_no_match_log() -> set[str]:
    """Load shortcodes of previously visited posts with no boulder match."""
    if not _NO_MATCH_LOG.is_file():
        return set()
    codes: set[str] = set()
    for line in _NO_MATCH_LOG.read_text(encoding="utf-8").splitlines():
        token = line.strip().split()[0] if line.strip() else ""
        if token:
            codes.add(token)
    return codes


def _append_no_match_log(shortcode: str, post_url: str) -> None:
    """Log a visited post that had no boulder match so we skip it next time."""
    _NO_MATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_NO_MATCH_LOG, "a", encoding="utf-8") as f:
        f.write(f"{shortcode} {post_url}\n")


def _next_boulder_folder(output_dir: Path, folder_name: str) -> Path:
    """Return the next available folder for *folder_name*.

    If ``output_dir/folder_name`` doesn't exist yet, return it as-is.
    Otherwise append ``_2``, ``_3``, … until an unused name is found.
    """
    candidate = output_dir / folder_name
    if not candidate.exists():
        return candidate
    m = 2
    while True:
        candidate = output_dir / f"{folder_name}_{m}"
        if not candidate.exists():
            return candidate
        m += 1

# Instagram CDN hostname fragments — used to recognise video URLs
_CDN_PATTERNS = ("cdninstagram", "fbcdn.net", "scontent")


def _shortcode_to_media_id(shortcode: str) -> int:
    """Convert an Instagram shortcode to a numeric media ID."""
    alphabet = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    )
    media_id = 0
    for char in shortcode:
        media_id = media_id * 64 + alphabet.index(char)
    return media_id


def _is_video_cdn_url(url: str) -> bool:
    """Return True if *url* looks like an Instagram video CDN URL."""
    if not any(p in url for p in _CDN_PATTERNS):
        return False
    return ".mp4" in url or "/v/t50" in url or "video" in url


# ── Download helper ──────────────────────────────────────────────────

def _download_video_from_page(
    pg: Page, dest_dir: Path, *, precaptured_urls: list[str] | None = None,
) -> Path | None:
    """
    Extract the video URL from the current page and download it.

    Instagram videos use blob: URLs in the <video> DOM elements, so we
    must find the real CDN URL from the page's embedded JSON data or
    by intercepting the GraphQL/XHR responses.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    output_path = dest_dir / "video.mp4"
    if output_path.exists():
        return output_path

    video_url = None

    # Strategy 1: Scan ALL script tags and page HTML for CDN video URLs.
    # Instagram embeds video_url in <script type="application/json"> data
    # and also in __additionalDataLoaded / require("ScheduledServerJS")
    # NOTE: This is tried FIRST because it's the most reliable for posts
    # that embed the URL.  Pre-captured URLs (from navigation network
    # traffic) are deferred because they often match tiny thumbnails.
    if not video_url:
        video_url = pg.evaluate(r"""
        () => {
            // Regex to find Instagram CDN video URLs (cdninstagram OR fbcdn)
            const cdnRe = /https?:\/\/[^\s"'\\]*?(?:cdninstagram|fbcdn)[^\s"'\\]*?\.mp4[^\s"'\\]*/gi;

            // Also match video_url JSON keys
            const jsonUrlRe = /"video_url"\s*:\s*"(https?:[^"]+)"/gi;

            // Also look for video_versions URLs
            const versionsRe = /"url"\s*:\s*"(https?:[^"]*(?:cdninstagram|fbcdn)[^"]*\.mp4[^"]*)"/gi;

            // Search all script tags (application/json, text/javascript, etc.)
            const scripts = document.querySelectorAll('script');
            for (const s of scripts) {
                const txt = s.textContent || '';
                if (!txt) continue;

                // Look for "video_url":"https://..."
                let m = jsonUrlRe.exec(txt);
                if (m) {
                    // Unescape JSON unicode escapes
                    const url = m[1].replace(/\\u0026/g, '&').replace(/\\\//g, '/');
                    if (url.startsWith('http')) return url;
                }

                // Look for CDN URLs containing cdninstagram/fbcdn and .mp4
                let m2 = cdnRe.exec(txt);
                if (m2) {
                    const url = m2[0].replace(/\\u0026/g, '&').replace(/\\\//g, '/');
                    if (url.startsWith('http')) return url;
                }

                // Look for video_versions URL entries
                let m3 = versionsRe.exec(txt);
                if (m3) {
                    const url = m3[1].replace(/\\u0026/g, '&').replace(/\\\//g, '/');
                    if (url.startsWith('http')) return url;
                }
            }

            // Try og:video meta tags
            for (const prop of ['og:video', 'og:video:secure_url']) {
                const el = document.querySelector(`meta[property="${prop}"]`);
                if (el) {
                    const u = el.getAttribute('content');
                    if (u && u.startsWith('http')) return u;
                }
            }

            return null;
        }
    """)

    if video_url:
        print(f"    [download] Found URL in page data")

    # Strategy 1b: Use video URLs captured during initial page navigation
    # (only those containing .mp4 — avoid thumbnail/image CDN hits).
    if not video_url and precaptured_urls:
        mp4_urls = [u for u in precaptured_urls if ".mp4" in u]
        if mp4_urls:
            video_url = mp4_urls[0]
            print(f"    [download] Using .mp4 URL captured during page load")

    if not video_url:
        # Strategy 2: Trigger video playback to force the browser to
        # fetch the actual video data from Instagram's CDN, then capture
        # the CDN URL from the network traffic.
        print("    [download] No URL in page data, triggering playback …")
        captured: list[str] = []

        def _on_video_resp(response):
            url = response.url
            ct = response.headers.get("content-type", "")
            # Direct video content-type (progressive download)
            if ct.startswith("video/"):
                captured.append(url)
                return
            # CDN URL patterns for Instagram video assets
            if _is_video_cdn_url(url):
                captured.append(url)
                return
            # JSON/JS responses that might embed a video_url
            if ("json" in ct or "javascript" in ct) and not captured:
                try:
                    body = response.text()
                    for m in re.finditer(
                        r'"video_url"\s*:\s*"(https?:[^"]+)"', body,
                    ):
                        vu = (
                            m.group(1)
                            .replace("\\u0026", "&")
                            .replace("\\/", "/")
                        )
                        captured.append(vu)
                except Exception:
                    pass

        pg.on("response", _on_video_resp)

        # Click the video element to kick off playback / buffering
        try:
            vid = pg.locator("video")
            if vid.count() > 0:
                vid.first.click(force=True)
        except Exception:
            pass

        # Wait up to 10 s for a CDN response to appear
        for _ in range(10):
            if captured:
                break
            time.sleep(1)

        try:
            pg.remove_listener("response", _on_video_resp)
        except Exception:
            pass

        if captured:
            video_url = captured[0]
            print("    [download] Captured CDN URL during playback")

    if not video_url:
        # Strategy 3: Instagram private media API.  Convert the post's
        # shortcode to a numeric media-ID and hit the v1/media endpoint
        # using the browser context's authenticated session cookies.
        current_url = pg.url
        shortcode_match = re.search(
            r"/(?:p|reel|tv)/([A-Za-z0-9_-]+)", current_url,
        )
        if shortcode_match:
            shortcode = shortcode_match.group(1)
            media_id = _shortcode_to_media_id(shortcode)
            print(f"    [download] Trying media API (ID={media_id}) …")

            api_url = (
                f"https://i.instagram.com/api/v1/media/{media_id}/info/"
            )
            # Grab CSRF token from the browser cookies
            csrf = ""
            for ck in pg.context.cookies():
                if ck["name"] == "csrftoken":
                    csrf = ck["value"]
                    break

            try:
                resp = pg.context.request.get(
                    api_url,
                    headers={
                        "x-ig-app-id": "936619743392459",
                        "x-requested-with": "XMLHttpRequest",
                        "x-csrftoken": csrf,
                        "user-agent": (
                            "Instagram 275.0.0.27.98 Android "
                            "(33/13; 420dpi; 1080x2400; samsung; "
                            "SM-G991B; o1s; exynos2100)"
                        ),
                    },
                    timeout=15000,
                )
                if resp.ok:
                    data = json.loads(resp.text())
                    for item in data.get("items", []):
                        versions = item.get("video_versions", [])
                        if versions:
                            video_url = versions[0].get("url")
                            if video_url:
                                print("    [download] Found URL via media API")
                                break
                        if item.get("video_url"):
                            video_url = item["video_url"]
                            print("    [download] Found URL via media API")
                            break
                else:
                    print(f"    [download] Media API HTTP {resp.status}")
            except Exception as e:
                print(f"    [download] Media API error: {e}")

    if not video_url:
        print("    [download] Could not find video URL")
        return None

    _MIN_VIDEO_BYTES = 50_000  # 50 KB — anything smaller is not a video

    # Download the video using the browser's authenticated session
    try:
        response = pg.context.request.get(video_url, timeout=60000)
        if response.ok:
            body = response.body()
            if len(body) < _MIN_VIDEO_BYTES:
                print(f"    [download] {len(body) / 1024:.0f} KB — too small, skipping")
            else:
                output_path.write_bytes(body)
                print(f"    [download] {len(body) / 1024:.0f} KB")
                return output_path
        else:
            print(f"    [download] HTTP {response.status}")
    except Exception as e:
        print(f"    [download] browser request failed: {e}")

    # Fallback: try urllib (CDN URLs are sometimes publicly accessible)
    try:
        urllib.request.urlretrieve(video_url, str(output_path))
        if output_path.exists():
            sz = output_path.stat().st_size
            if sz >= _MIN_VIDEO_BYTES:
                print(f"    [download] {sz / 1024:.0f} KB (urllib)")
                return output_path
            else:
                output_path.unlink(missing_ok=True)
                print(f"    [download] {sz / 1024:.0f} KB (urllib) — too small")
    except Exception:
        pass

    return None


# ── Login helpers ────────────────────────────────────────────────────

def _dismiss_cookie_banner(pg: Page) -> None:
    """Try to dismiss Instagram's cookie-consent overlay."""
    consent_texts = [
        "Allow essential and optional cookies",
        "Allow all cookies",
        "Accept all",
        "Accept",
        "Allow",
        "Decline optional cookies",
        "Only allow essential cookies",
    ]
    for txt in consent_texts:
        try:
            btn = pg.locator(f"button:has-text('{txt}')")
            if btn.count() > 0:
                btn.first.click()
                time.sleep(1)
                return
        except Exception:
            pass

    # Fallback: click any button inside a role=dialog
    try:
        dialog_btns = pg.locator('div[role="dialog"] button')
        if dialog_btns.count() > 0:
            dialog_btns.first.click()
            time.sleep(1)
    except Exception:
        pass


def _find_element(pg: Page, selectors: list[str]):
    """Return the first matching locator from a list of CSS selectors."""
    for sel in selectors:
        try:
            loc = pg.locator(sel)
            if loc.count() > 0:
                return loc.first
        except Exception:
            pass
    return None


def _do_login(pg: Page, username: str, password: str) -> bool:
    """Fill in the Instagram login form. Returns True on success."""
    _dismiss_cookie_banner(pg)
    time.sleep(1)

    user_input = _find_element(pg, [
        'input[name="username"]',
        'input[aria-label*="username"]',
        'input[aria-label*="email"]',
        'input[aria-label*="Mobile"]',
        'input[type="text"]',
    ])
    pass_input = _find_element(pg, [
        'input[name="password"]',
        'input[type="password"]',
        'input[aria-label*="assword"]',
    ])

    if not user_input or not pass_input:
        print("[browser] Could not find login form inputs.")
        return False

    print(f"[browser] Logging in as {username} …")
    user_input.click()
    time.sleep(0.3)
    user_input.fill(username)
    time.sleep(0.3)
    pass_input.click()
    time.sleep(0.3)
    pass_input.fill(password)
    time.sleep(0.3)

    # Click login button
    login_btn = _find_element(pg, [
        'button[type="submit"]',
        'button:has-text("Log in")',
        'div[role="button"]:has-text("Log in")',
    ])
    if login_btn:
        login_btn.click()
    else:
        pg.keyboard.press("Enter")

    # Wait for navigation away from login page
    try:
        pg.wait_for_url(lambda u: "/accounts/login" not in u, timeout=20000)
    except Exception:
        print("[browser] Login may have failed — still on login page.")
        return False

    time.sleep(3)

    # Handle "Save login info?" and notification prompts
    for _ in range(3):
        try:
            not_now = pg.locator(
                "button:has-text('Not Now'), button:has-text('Not now')"
            )
            if not_now.count() > 0:
                not_now.first.click()
                time.sleep(2)
        except Exception:
            pass

    print("[browser] Logged in.")
    return True


# ── Caption extraction ───────────────────────────────────────────────

def _extract_caption(pg: Page) -> str:
    """Try several strategies to pull the actual post caption text."""

    # Strategy 1: JavaScript — walk the DOM to find the caption.
    # Instagram wraps the caption in an <h1> inside the first <li> of a <ul>
    # within the <article>, or in a span near the username link.
    try:
        text = pg.evaluate("""
            () => {
                // Method A: <h1> inside article (most common for post pages)
                const h1 = document.querySelector('article h1');
                if (h1 && h1.innerText.trim().length > 5) {
                    return h1.innerText.trim();
                }

                // Method B: first <li> span text that is long enough
                const items = document.querySelectorAll('article ul li');
                for (const li of items) {
                    const spans = li.querySelectorAll('span');
                    for (const s of spans) {
                        const t = s.innerText.trim();
                        if (t.length > 20) return t;
                    }
                }

                // Method C: entire article text (last resort)
                const article = document.querySelector('article');
                if (article) return article.innerText.substring(0, 3000);

                return '';
            }
        """)
        if text and len(text) > 5:
            return text
    except Exception:
        pass

    # Strategy 2: meta og:description — contains IG summary which sometimes
    # includes the caption after the "likes" prefix.  Parse out the quoted part.
    try:
        meta = pg.locator('meta[property="og:description"]')
        if meta.count() > 0:
            content = meta.get_attribute("content", timeout=3000) or ""
            # Format: '123 likes, 4 comments - user on date: "actual caption..."'
            m = re.search(r': ["\u201c](.+)["\u201d]', content)
            if m:
                return m.group(1)
            if content and len(content) > 10:
                return content
    except Exception:
        pass

    return ""


# ── Main scraper ─────────────────────────────────────────────────────

def scrape_hashtag_browser(
    hashtag: str = "castlehillbouldering",
    username: str | None = None,
    password: str | None = None,
    max_videos: int = 5,
    output_dir: Path = SCRAPE_DIR,
    headless: bool = False,
) -> list[dict]:
    """
    Scrape Instagram hashtag page using a real browser.

    Parameters
    ----------
    hashtag : str
        Hashtag to search (without #).
    username, password : str
        Instagram credentials.
    max_videos : int
        Stop after downloading this many matching videos.
    output_dir : Path
        Where to save scraped data.
    headless : bool
        Run browser without GUI (set False to watch/debug).

    Returns
    -------
    list[dict]
        Downloaded video metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    already_downloaded = _load_url_log()
    no_match_codes = _load_no_match_log()
    already_downloaded |= no_match_codes
    if already_downloaded:
        print(f"[browser] URL log: {len(already_downloaded)} previously visited posts "
              f"({len(no_match_codes)} no-match)")

    with sync_playwright() as p:
        _SESSION_DIR.mkdir(parents=True, exist_ok=True)
        browser = p.chromium.launch_persistent_context(
            user_data_dir=str(_SESSION_DIR),
            headless=headless,
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = browser.pages[0] if browser.pages else browser.new_page()

        # ── Navigate to the hashtag page ─────────────────────────
        hashtag_url = f"https://www.instagram.com/explore/tags/{hashtag}/"
        print(f"[browser] Navigating to #{hashtag} …")
        page.goto(hashtag_url, wait_until="domcontentloaded", timeout=30000)
        time.sleep(5)

        # Check if we got redirected to login
        if "/accounts/login" in page.url:
            print("[browser] Redirected to login — session expired or missing.")
            if not username or not password:
                print("[browser] No credentials provided. Aborting.")
                browser.close()
                return []
            if not _do_login(page, username, password):
                browser.close()
                return []
            # Re-navigate to hashtag page after login
            print(f"[browser] Re-navigating to #{hashtag} …")
            page.goto(hashtag_url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(5)

        # Verify we're not stuck on login
        current = page.url
        print(f"[browser] Current URL: {current}")
        if "/accounts/login" in current:
            print("[browser] Still on login page — aborting.")
            browser.close()
            return []

        # ── Collect post links ───────────────────────────────────
        print("[browser] Scanning posts …")
        seen_codes: set[str] = set()
        scroll_attempts = 0
        max_scrolls = 20

        while len(results) < max_videos and scroll_attempts < max_scrolls:
            # Use JavaScript to gather all post links on the page
            post_hrefs: list[str] = page.evaluate("""
                () => {
                    const links = document.querySelectorAll('a');
                    const found = [];
                    links.forEach(a => {
                        const href = a.getAttribute('href') || '';
                        if (href.match(/\\/(p|reel)\\/[A-Za-z0-9_-]+/)) {
                            found.push(href);
                        }
                    });
                    return found;
                }
            """)

            new_codes: list[tuple[str, str]] = []
            for href in post_hrefs:
                m = re.search(r'/(?:p|reel)/([A-Za-z0-9_-]+)', href)
                if m and m.group(1) not in seen_codes:
                    sc = m.group(1)
                    seen_codes.add(sc)
                    if sc in already_downloaded:
                        continue  # skip — already in URL log
                    new_codes.append((sc, href))

            if new_codes:
                print(f"  Found {len(new_codes)} new posts "
                      f"(total seen: {len(seen_codes)})")

            # Visit each new post to check caption + video
            for shortcode, href in new_codes:
                if len(results) >= max_videos:
                    break

                post_url = (f"https://www.instagram.com{href}"
                            if href.startswith("/") else href)

                try:
                    # Set up response interception BEFORE navigation
                    # to catch video URLs from Instagram's XHR/fetch calls
                    captured_video_urls = []

                    def on_post_response(response):
                        url = response.url
                        ct = response.headers.get("content-type", "")
                        # Catch direct video content-type responses
                        if ct.startswith("video/"):
                            captured_video_urls.append(url)
                            return
                        # Catch JSON responses with video_url
                        if ("json" in ct or "javascript" in ct):
                            try:
                                body = response.text()
                                for m in re.finditer(
                                    r'"video_url"\s*:\s*"(https?:[^"]+)"',
                                    body,
                                ):
                                    vu = (
                                        m.group(1)
                                        .replace("\\u0026", "&")
                                        .replace("\\/", "/")
                                    )
                                    captured_video_urls.append(vu)
                            except Exception:
                                pass

                    page.on("response", on_post_response)
                    page.goto(post_url, wait_until="domcontentloaded",
                              timeout=15000)
                    time.sleep(3)
                    page.remove_listener("response", on_post_response)

                    # Extract caption
                    caption = _extract_caption(page)

                    # Check for boulder match
                    boulders = find_boulders(caption)
                    is_video = page.locator("video").count() > 0

                    status = "VIDEO" if is_video else "photo"
                    match_str = (", ".join(b.label for b in boulders)
                                 if boulders else "no match")
                    cap_preview = (caption[:60].replace("\n", " ")
                                   if caption else "(no caption)")
                    print(f"  [{len(seen_codes)}] {shortcode} [{status}] "
                          f"[{match_str}] {cap_preview}")

                    if boulders and is_video:
                        # ── Multi-boulder → log for future, don't download
                        if len(boulders) > 1:
                            names = ", ".join(b.label for b in boulders)
                            print(f"    [multi] {len(boulders)} boulders "
                                  f"({names}) — logging for future")
                            _append_multi_boulder_log(
                                shortcode, post_url, boulders)
                            if shortcode not in already_downloaded:
                                _append_url_log(shortcode, post_url)
                                already_downloaded.add(shortcode)
                        else:
                            # ── Single boulder — download ────────
                            b = boulders[0]
                            boulder_dir = _next_boulder_folder(
                                output_dir, b.folder_name)
                            dest = boulder_dir / shortcode
                            dest.mkdir(parents=True, exist_ok=True)
                            (dest / "caption.txt").write_text(
                                caption, encoding="utf-8"
                            )

                            print(f"    -> Downloading {b.label} …")
                            video_path = _download_video_from_page(
                                page, dest,
                                precaptured_urls=captured_video_urls,
                            )

                            if video_path:
                                print(f"    OK saved to "
                                      f"{dest.relative_to(output_dir)}")
                                results.append({
                                    "shortcode": shortcode,
                                    "boulder": b.name,
                                    "grade": b.grade,
                                    "folder": str(dest),
                                    "caption": caption[:200],
                                })
                                if shortcode not in already_downloaded:
                                    _append_url_log(shortcode, post_url)
                                    already_downloaded.add(shortcode)
                            else:
                                print("    FAIL download failed")

                    else:
                        # No matching boulder or not a video — log so we skip next time
                        _append_no_match_log(shortcode, post_url)
                        already_downloaded.add(shortcode)

                except Exception as exc:
                    print(f"  [{len(seen_codes)}] {shortcode} — error: {exc}")

                time.sleep(1)

            if len(results) >= max_videos:
                break

            # Navigate back to hashtag page and scroll further
            try:
                page.goto(hashtag_url, wait_until="domcontentloaded",
                          timeout=30000)
            except Exception:
                print("[browser] Lost page — stopping.")
                break
            time.sleep(3)
            for _ in range(scroll_attempts + 2):
                page.keyboard.press("End")
                time.sleep(1.5)
            scroll_attempts += 1

        try:
            browser.close()
        except Exception:
            pass

    print(f"\n[browser] Done. Scanned {len(seen_codes)} posts, "
          f"downloaded {len(results)} matching videos.\n")
    return results


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Browser-based Instagram scraper"
    )
    parser.add_argument("--hashtag", default="castlehillbouldering")
    parser.add_argument("--max-videos", type=int, default=5)
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--output-dir", default=str(SCRAPE_DIR))
    parser.add_argument("--headless", action="store_true",
                        help="Run browser without GUI")
    args = parser.parse_args()

    # Load .env
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.is_file():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                k, _, v = line.partition("=")
                if k.strip():
                    os.environ.setdefault(k.strip(), v.strip())

    username = args.username or os.environ.get("IG_USERNAME")
    password = args.password or os.environ.get("IG_PASSWORD")

    scrape_hashtag_browser(
        hashtag=args.hashtag,
        username=username,
        password=password,
        max_videos=args.max_videos,
        output_dir=Path(args.output_dir),
        headless=args.headless,
    )
