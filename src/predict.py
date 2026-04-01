"""
Classify new images using a trained boulder classifier model.

Usage:
    python src/predict.py path/to/image.jpg
    python src/predict.py path/to/folder_of_images/
"""

import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from prepare_data import IMG_SIZE, VALID_EXTENSIONS


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "boulder_classifier.pth"
META_PATH = MODEL_DIR / "model_meta.json"

# Same normalisation used during training
predict_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model() -> tuple[nn.Module, list[str]]:
    """Load the trained model and class names."""
    if not META_PATH.exists():
        print(f"[ERROR] No metadata found at {META_PATH}")
        print("Run `python src/train.py` first.")
        sys.exit(1)

    with open(META_PATH) as f:
        meta = json.load(f)

    class_names = meta["classes"]
    num_classes = meta["num_classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, num_classes),
    )

    if not MODEL_PATH.exists():
        print(f"[ERROR] No trained model found at {MODEL_PATH}")
        print("Run `python src/train.py` first.")
        sys.exit(1)

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model, class_names


def predict_single(img_path: Path, model: nn.Module,
                   class_names: list[str]) -> tuple[str, float]:
    """Return (predicted_class, confidence) for a single image."""
    img = Image.open(img_path).convert("RGB")
    tensor = predict_transforms(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(probs.argmax())
    return class_names[pred_idx], float(probs[pred_idx])


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_or_folder>")
        sys.exit(1)

    target = Path(sys.argv[1])

    print("Loading model …")
    model, class_names = load_model()
    print(f"  {len(class_names)} classes: {', '.join(class_names)}\n")

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
    print(f"Classifying {len(paths)} image(s):\n")
    print(f"  {'File':<40s}  {'Prediction':<25s}  {'Confidence':>10s}")
    print(f"  {'-'*40}  {'-'*25}  {'-'*10}")

    for p in paths:
        try:
            label, conf = predict_single(p, model, class_names)
            display_label = label.replace("_", " ")
            print(f"  {p.name:<40s}  {display_label:<25s}  {conf:>9.1%}")
        except Exception as e:
            print(f"  {p.name:<40s}  ERROR: {e}")

    print()


if __name__ == "__main__":
    main()
