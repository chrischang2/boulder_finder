"""
Data preparation utilities for the Boulder Finder image classifier.

Expected directory layout:
    data/
        train/
            boulder/        ← images of boulders
            not_boulder/    ← images that are NOT boulders
        test/
            boulder/
            not_boulder/

Place your images in the folders above, then run this script to verify
the dataset and generate a summary.
"""

import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
from skimage.io import imread
from skimage.transform import resize


# ── constants ────────────────────────────────────────────────────────
IMG_SIZE = (128, 128)          # all images resized to this for training
CLASSES = ["boulder", "not_boulder"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ── helpers ──────────────────────────────────────────────────────────
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXTENSIONS


def load_image(path: Path, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Load an image, convert to RGB, resize, and return as float32 array."""
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_dataset(split: str = "train") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load all images for a given split ('train' or 'test').

    Returns
    -------
    X : ndarray of shape (n_samples, height*width*3)
        Flattened, normalised pixel data.
    y : ndarray of shape (n_samples,)
        Integer labels (0 = boulder, 1 = not_boulder).
    filenames : list[str]
        Original file paths (useful for debugging).
    """
    split_dir = DATA_DIR / split
    if not split_dir.exists():
        print(f"[ERROR] Split directory not found: {split_dir}")
        sys.exit(1)

    images, labels, filenames = [], [], []

    for label_idx, class_name in enumerate(CLASSES):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            print(f"[WARN] Class directory missing — creating: {class_dir}")
            class_dir.mkdir(parents=True, exist_ok=True)
            continue

        for img_path in sorted(class_dir.iterdir()):
            if not _is_image(img_path):
                continue
            try:
                img = load_image(img_path)
                images.append(img.flatten())
                labels.append(label_idx)
                filenames.append(str(img_path))
            except Exception as e:
                print(f"[WARN] Skipping {img_path.name}: {e}")

    if not images:
        print(f"[ERROR] No images found in {split_dir}. "
              f"Add images to {'/'.join(str(c) for c in CLASSES)} sub-folders.")
        sys.exit(1)

    X = np.stack(images)
    y = np.array(labels)
    return X, y, filenames


def dataset_summary(split: str = "train") -> None:
    """Print a summary of the dataset for a given split."""
    split_dir = DATA_DIR / split
    print(f"\n{'='*40}")
    print(f"  Dataset summary  —  split: {split}")
    print(f"{'='*40}")

    total = 0
    for class_name in CLASSES:
        class_dir = split_dir / class_name
        if class_dir.exists():
            count = sum(1 for f in class_dir.iterdir() if _is_image(f))
        else:
            count = 0
        print(f"  {class_name:20s}: {count:>5} images")
        total += count

    print(f"  {'TOTAL':20s}: {total:>5} images")
    print(f"{'='*40}\n")


# ── CLI entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    # Create placeholder class folders if they don't exist
    for split in ("train", "test"):
        for cls in CLASSES:
            (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    print("Directory structure is ready. Add your images to:")
    for split in ("train", "test"):
        for cls in CLASSES:
            print(f"  {DATA_DIR / split / cls}")

    # Show summary for any images already present
    for split in ("train", "test"):
        dataset_summary(split)
