"""
Extract frames from downloaded boulder videos.

For each video.mp4 found under data/scraped/, extract frames at a
configurable interval and save them as JPEGs in a ``frames/`` sub-folder
next to the video.

Uses OpenCV (cv2) for decoding.

Layout after extraction:
    data/scraped/<boulder>/<shortcode>/
        video.mp4
        caption.txt
        frames/
            frame_0001.jpg
            frame_0002.jpg
            …
"""

from __future__ import annotations

from pathlib import Path

import cv2  # type: ignore

SCRAPE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "scraped"


def extract_frames(
    video_path: Path,
    output_dir: Path | None = None,
    every_n_seconds: float = 1.0,
    max_frames: int = 300,
    quality: int = 95,
) -> list[Path]:
    """
    Extract frames from *video_path* and save as JPEGs.

    Parameters
    ----------
    video_path : Path
        Path to the .mp4 file.
    output_dir : Path | None
        Folder to save frames into.  Defaults to ``<video_dir>/frames/``.
    every_n_seconds : float
        Interval between extracted frames (in seconds).
    max_frames : int
        Safety cap on the number of frames extracted.
    quality : int
        JPEG quality (1–100).

    Returns
    -------
    list[Path]
        Paths to the saved frame images.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_dir is None:
        output_dir = video_path.parent / "frames"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps else 0

    frame_interval = max(1, int(fps * every_n_seconds))

    saved: list[Path] = []
    frame_idx = 0
    save_count = 0

    print(f"  [frames] {video_path.name}: {total_frames} frames, "
          f"{fps:.1f} fps, ~{duration_s:.1f}s — extracting every {every_n_seconds}s")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0 and save_count < max_frames:
            save_count += 1
            fname = output_dir / f"frame_{save_count:04d}.jpg"
            cv2.imwrite(
                str(fname),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, quality],
            )
            saved.append(fname)

        frame_idx += 1

    cap.release()
    print(f"  [frames] Saved {len(saved)} frames → {output_dir}")
    return saved


def extract_all(
    scrape_root: Path = SCRAPE_DIR,
    every_n_seconds: float = 1.0,
    skip_existing: bool = True,
) -> int:
    """
    Walk the scraped directory tree and extract frames from every video.

    Parameters
    ----------
    scrape_root : Path
        Root of the scraped data (default ``data/scraped/``).
    every_n_seconds : float
        Frame extraction interval.
    skip_existing : bool
        If True, skip videos that already have a ``frames/`` folder
        with at least one image.

    Returns
    -------
    int
        Number of videos processed.
    """
    processed = 0
    for video_path in sorted(scrape_root.rglob("video.mp4")):
        frames_dir = video_path.parent / "frames"

        if skip_existing and frames_dir.exists():
            existing = list(frames_dir.glob("*.jpg"))
            if existing:
                print(f"  [skip] {video_path.parent.name} already has "
                      f"{len(existing)} frames")
                continue

        extract_frames(video_path, every_n_seconds=every_n_seconds)
        processed += 1

    return processed


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract frames from scraped boulder videos."
    )
    parser.add_argument("--input", default=str(SCRAPE_DIR),
                        help="Root scraped folder (default: data/scraped/)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between frames (default: 1.0)")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if frames already exist")
    args = parser.parse_args()

    n = extract_all(
        scrape_root=Path(args.input),
        every_n_seconds=args.interval,
        skip_existing=not args.force,
    )
    print(f"\nProcessed {n} videos.")
