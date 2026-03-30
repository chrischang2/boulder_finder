"""
Classify new images using a trained boulder classifier model.

Usage:
    python src/predict.py path/to/image.jpg
    python src/predict.py path/to/folder_of_images/
"""

import sys
from pathlib import Path

import numpy as np
import joblib
from PIL import Image

from prepare_data import CLASSES, IMG_SIZE, VALID_EXTENSIONS, load_image
from train import extract_features


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "boulder_classifier.joblib"


def predict_single(img_path: Path, model) -> tuple[str, float]:
    """Return (predicted_class, confidence) for a single image."""
    img = load_image(img_path)
    X = extract_features(img.flatten().reshape(1, -1))
    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    return CLASSES[pred_idx], float(proba[pred_idx])


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_or_folder>")
        sys.exit(1)

    target = Path(sys.argv[1])

    if not MODEL_PATH.exists():
        print(f"[ERROR] No trained model found at {MODEL_PATH}")
        print("Run `python src/train.py` first.")
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH} …")
    model = joblib.load(MODEL_PATH)

    # Collect image paths
    if target.is_dir():
        paths = sorted(
            p for p in target.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        )
        if not paths:
            print(f"[ERROR] No images found in {target}")
            sys.exit(1)
    elif target.is_file():
        paths = [target]
    else:
        print(f"[ERROR] Path not found: {target}")
        sys.exit(1)

    # Predict
    print(f"\nClassifying {len(paths)} image(s):\n")
    print(f"  {'File':<40s}  {'Prediction':<15s}  {'Confidence':>10s}")
    print(f"  {'-'*40}  {'-'*15}  {'-'*10}")

    for p in paths:
        try:
            label, conf = predict_single(p, model)
            print(f"  {p.name:<40s}  {label:<15s}  {conf:>9.1%}")
        except Exception as e:
            print(f"  {p.name:<40s}  ERROR: {e}")

    print()


if __name__ == "__main__":
    main()
