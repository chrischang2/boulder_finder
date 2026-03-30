"""
Train an image classifier that distinguishes boulders from non-boulders.

Uses scikit-learn with HOG features + an SVM, which works well for binary
image classification on small-to-medium datasets.

Usage:
    python src/train.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
from skimage.feature import hog
from PIL import Image

from prepare_data import load_dataset, CLASSES, IMG_SIZE, DATA_DIR


# ── Feature extraction ──────────────────────────────────────────────
def extract_hog_features(X_flat: np.ndarray) -> np.ndarray:
    """
    Extract HOG (Histogram of Oriented Gradients) features from flattened
    RGB images.  HOG captures edge / gradient structure and is far more
    informative than raw pixels for traditional ML classifiers.
    """
    h, w = IMG_SIZE
    features = []
    for row in X_flat:
        img = row.reshape(h, w, 3)
        # Convert to grayscale for HOG
        gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        feat = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        )
        features.append(feat)
    return np.array(features)


def extract_color_histogram(X_flat: np.ndarray, bins: int = 32) -> np.ndarray:
    """Extract per-channel colour histograms as features."""
    h, w = IMG_SIZE
    features = []
    for row in X_flat:
        img = row.reshape(h, w, 3)
        hist_r = np.histogram(img[:, :, 0], bins=bins, range=(0, 1))[0]
        hist_g = np.histogram(img[:, :, 1], bins=bins, range=(0, 1))[0]
        hist_b = np.histogram(img[:, :, 2], bins=bins, range=(0, 1))[0]
        features.append(np.concatenate([hist_r, hist_g, hist_b]))
    return np.array(features, dtype=np.float32)


def extract_features(X_flat: np.ndarray) -> np.ndarray:
    """Combine HOG + colour histogram features."""
    print("  Extracting HOG features …")
    hog_feats = extract_hog_features(X_flat)
    print(f"    HOG shape: {hog_feats.shape}")

    print("  Extracting colour histogram features …")
    color_feats = extract_color_histogram(X_flat)
    print(f"    Colour hist shape: {color_feats.shape}")

    combined = np.hstack([hog_feats, color_feats])
    print(f"  Combined feature vector length: {combined.shape[1]}")
    return combined


# ── Training ─────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def train(model_name: str = "svm") -> None:
    # ── Load data ────────────────────────────────────────────────────
    print("\n[1/4] Loading training data …")
    X_train_raw, y_train, _ = load_dataset("train")
    print(f"  Loaded {len(y_train)} training images.")

    print("\n[2/4] Extracting features …")
    X_train = extract_features(X_train_raw)

    # ── Select model ─────────────────────────────────────────────────
    print(f"\n[3/4] Training {model_name} classifier …")
    t0 = time.time()

    if model_name == "svm":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
        ])
    elif model_name == "rf":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=None, random_state=42)),
        ])
    elif model_name == "gb":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=5, random_state=42)),
        ])
    else:
        print(f"Unknown model: {model_name}. Choose svm, rf, or gb.")
        sys.exit(1)

    # Cross-validation on training set
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    print(f"  5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Fit on full training set
    pipe.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Training completed in {elapsed:.1f}s.")

    # ── Evaluate on test set (if available) ──────────────────────────
    test_dir = DATA_DIR / "test"
    has_test = any(
        (test_dir / cls).exists() and any((test_dir / cls).iterdir())
        for cls in CLASSES
        if (test_dir / cls).exists()
    )

    if has_test:
        print("\n[4/4] Evaluating on test set …")
        X_test_raw, y_test, _ = load_dataset("test")
        X_test = extract_features(X_test_raw)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"\n  Test accuracy: {acc:.3f}")
        print("\n  Classification report:")
        print(classification_report(
            y_test, y_pred, target_names=CLASSES, digits=3))

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix — Test Set")
        cm_path = MODEL_DIR / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        print(f"  Confusion matrix saved to: {cm_path}")
        plt.close()
    else:
        print("\n[4/4] No test images found — skipping evaluation.")

    # ── Save model ───────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "boulder_classifier.joblib"
    joblib.dump(pipe, model_path)
    print(f"\n  Model saved to: {model_path}")
    print("  Done!\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the boulder classifier.")
    parser.add_argument(
        "--model", choices=["svm", "rf", "gb"], default="svm",
        help="Model type: svm (default), rf (Random Forest), gb (Gradient Boosting).",
    )
    args = parser.parse_args()
    train(model_name=args.model)
