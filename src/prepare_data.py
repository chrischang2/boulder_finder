"""
Data preparation utilities for the Boulder Finder image classifier.

Supports two layouts:

1. **Standard** (``data/train/<class>/*.jpg``):
       data/train/cyclops/img1.jpg
       data/train/the_joker/img2.jpg

2. **Scraped** (``data/scraped/<class>/<shortcode>/frames/*.jpg``):
       data/scraped/vlad_the_impaler/BqWxA5EBa7b/frames/frame_0001.jpg

Each sub-folder name becomes a class label.
"""

import os
import sys
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


# ── constants ────────────────────────────────────────────────────────
IMG_SIZE = 224                 # standard ResNet input size
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SCRAPED_DIR = DATA_DIR / "scraped"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ── transforms ───────────────────────────────────────────────────────
# Heavy augmentation is critical with very few images per class.
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_dataset(split: str = "train") -> datasets.ImageFolder:
    """
    Load an ImageFolder dataset for the given split.

    Returns a torchvision ImageFolder whose classes are auto-discovered
    from the sub-folder structure.
    """
    split_dir = DATA_DIR / split
    if not split_dir.exists():
        print(f"[ERROR] Split directory not found: {split_dir}")
        sys.exit(1)

    tfm = train_transforms if split == "train" else test_transforms
    dataset = datasets.ImageFolder(str(split_dir), transform=tfm)

    if len(dataset) == 0:
        print(f"[ERROR] No images found in {split_dir}.")
        sys.exit(1)

    return dataset


def dataset_summary(split: str = "train") -> None:
    """Print a summary of the dataset for a given split."""
    split_dir = DATA_DIR / split
    if not split_dir.exists():
        print(f"  Split '{split}' directory not found.")
        return

    print(f"\n{'='*50}")
    print(f"  Dataset summary  —  split: {split}")
    print(f"{'='*50}")

    total = 0
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        count = sum(1 for f in class_dir.iterdir()
                    if f.suffix.lower() in VALID_EXTENSIONS)
        if count > 0:
            print(f"  {class_dir.name:30s}: {count:>3} images")
            total += count

    print(f"  {'TOTAL':30s}: {total:>3} images")
    print(f"{'='*50}\n")


# ── Scraped-data dataset ────────────────────────────────────────────
class ScrapedDataset(Dataset):
    """Load frames from ``data/scraped/<class>/<shortcode>/frames/*.jpg``.

    Flattens the nested scraped layout into a standard (image, label)
    dataset usable by DataLoader.
    """

    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}

        # Discover classes (top-level sub-dirs that contain frames)
        class_dirs = sorted(
            d for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        idx = 0
        for class_dir in class_dirs:
            # Collect all jpg files recursively under this class
            imgs = sorted(
                str(f)
                for f in class_dir.rglob("*")
                if f.suffix.lower() in VALID_EXTENSIONS and f.is_file()
            )
            if not imgs:
                continue
            self.classes.append(class_dir.name)
            self.class_to_idx[class_dir.name] = idx
            for img_path in imgs:
                self.samples.append((img_path, idx))
            idx += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def load_scraped_dataset(transform=None):
    """Load the scraped dataset with the given transform."""
    if transform is None:
        transform = train_transforms
    ds = ScrapedDataset(SCRAPED_DIR, transform=transform)
    if len(ds) == 0:
        print(f"[ERROR] No scraped images found in {SCRAPED_DIR}.")
        sys.exit(1)
    return ds


# ── Test-set dataset (loose files, filename = class label) ──────────
TEST_DIR = DATA_DIR / "test"


class TestDataset(Dataset):
    """Load test images from ``data/test/``.

    Each file's stem (filename without extension) is sanitised
    (lowercased, spaces/dashes → underscores) to match the training
    class names.  Only images whose label appears in *known_classes*
    are included so the numeric label indices stay consistent.
    """

    def __init__(
        self,
        root: str | Path,
        known_classes: list[str],
        transform=None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.classes = known_classes
        self.class_to_idx = {c: i for i, c in enumerate(known_classes)}
        self.samples: list[tuple[str, int]] = []
        self.labels_text: list[str] = []  # human-readable per-sample

        import re
        sanitize = re.compile(r"[^a-z0-9_]+")

        for f in sorted(self.root.iterdir()):
            if not f.is_file() or f.suffix.lower() not in VALID_EXTENSIONS:
                continue
            raw_label = sanitize.sub("_", f.stem.lower()).strip("_")
            if raw_label in self.class_to_idx:
                self.samples.append((str(f), self.class_to_idx[raw_label]))
                self.labels_text.append(raw_label)
            else:
                print(f"  [test] WARNING: '{f.name}' -> '{raw_label}' "
                      f"not in training classes, skipping.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def load_test_dataset(known_classes: list[str], transform=None):
    """Load the hand-curated test set, filtering to *known_classes*."""
    if transform is None:
        transform = test_transforms
    if not TEST_DIR.exists():
        print(f"[WARN] Test directory not found: {TEST_DIR}")
        return None
    ds = TestDataset(TEST_DIR, known_classes, transform=transform)
    if len(ds) == 0:
        print("[WARN] No matching test images found.")
        return None
    return ds


def scraped_summary() -> None:
    """Print a summary of the scraped dataset."""
    if not SCRAPED_DIR.exists():
        print("  Scraped directory not found.")
        return
    print(f"\n{'='*50}")
    print(f"  Dataset summary  —  scraped data")
    print(f"{'='*50}")
    total = 0
    for class_dir in sorted(SCRAPED_DIR.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        count = sum(
            1 for f in class_dir.rglob("*")
            if f.suffix.lower() in VALID_EXTENSIONS and f.is_file()
        )
        if count > 0:
            print(f"  {class_dir.name:30s}: {count:>3} frames")
            total += count
    print(f"  {'TOTAL':30s}: {total:>3} frames")
    print(f"{'='*50}\n")


# ── CLI entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    for split in ("train", "test"):
        dataset_summary(split)
    scraped_summary()
