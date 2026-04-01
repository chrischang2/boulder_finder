"""
Train a multi-class boulder image classifier using transfer learning.

Uses a pre-trained ResNet-18, replacing the final FC layer for the
number of boulder classes discovered in data/scraped/.  Heavy data
augmentation compensates for the small per-class sample count.

After training, Grad-CAM heatmaps are generated for each class showing
which visual features the model considers most important.

Usage:
    py -3 src/train.py                        # default settings
    py -3 src/train.py --epochs 50            # more epochs
    py -3 src/train.py --unfreeze 30          # unfreeze backbone later
"""

import sys
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models

from prepare_data import (
    load_scraped_dataset,
    load_test_dataset,
    scraped_summary,
    train_transforms,
    test_transforms,
    IMG_SIZE,
    SCRAPED_DIR,
)


# ── paths ────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR = MODEL_DIR / "feature_maps"
VIS_DIR.mkdir(parents=True, exist_ok=True)


# ── model builder ────────────────────────────────────────────────────
def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """Return a ResNet-18 with a fresh classifier head."""
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Freeze backbone initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )

    return model.to(device)


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


# ── Grad-CAM ─────────────────────────────────────────────────────────
class GradCAM:
    """Compute Grad-CAM heatmaps for a target class."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None

        # Register hooks on target layer
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int):
        """Return a (H, W) heatmap as a numpy array in [0, 1]."""
        self.model.eval()
        output = self.model(input_tensor)

        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        # Global-average-pool the gradients -> channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def generate_feature_maps(
    model: nn.Module,
    dataset,
    class_names: list[str],
    device: torch.device,
    out_dir: Path,
    samples_per_class: int = 3,
) -> None:
    """Generate and save Grad-CAM heatmaps for each boulder class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from PIL import Image

    print("\n[viz] Generating Grad-CAM feature importance maps ...")

    # Target layer = last conv block of ResNet-18
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    # Group sample indices by class
    class_indices: dict[int, list[int]] = {}
    for i, (_, label) in enumerate(dataset.samples):
        class_indices.setdefault(label, []).append(i)

    for class_idx, class_name in enumerate(class_names):
        indices = class_indices.get(class_idx, [])
        if not indices:
            continue

        # Pick evenly-spaced samples from the class
        step = max(1, len(indices) // samples_per_class)
        picked = indices[::step][:samples_per_class]

        fig, axes = plt.subplots(
            len(picked), 3,
            figsize=(12, 4 * len(picked)),
            squeeze=False,
        )
        fig.suptitle(
            f"Feature Importance - {class_name.replace('_', ' ').title()}",
            fontsize=16, fontweight="bold", y=0.98,
        )

        for row, sample_idx in enumerate(picked):
            img_path = dataset.samples[sample_idx][0]
            # Load raw image for display
            raw_img = Image.open(img_path).convert("RGB").resize(
                (IMG_SIZE, IMG_SIZE)
            )

            # Load tensor with eval transforms
            tensor_img = test_transforms(raw_img).unsqueeze(0).to(device)

            # Compute Grad-CAM
            heatmap = grad_cam(tensor_img, class_idx)

            # Resize heatmap to image size
            heatmap_resized = np.array(
                Image.fromarray(
                    (heatmap * 255).astype(np.uint8)
                ).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            ) / 255.0

            # Create overlay
            raw_np = np.array(raw_img) / 255.0
            colormap = cm.jet(heatmap_resized)[:, :, :3]
            overlay = 0.5 * raw_np + 0.5 * colormap

            # Plot: original | heatmap | overlay
            axes[row][0].imshow(raw_np)
            axes[row][0].set_title("Original", fontsize=10)
            axes[row][0].axis("off")

            axes[row][1].imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
            axes[row][1].set_title("Grad-CAM Heatmap", fontsize=10)
            axes[row][1].axis("off")

            axes[row][2].imshow(np.clip(overlay, 0, 1))
            axes[row][2].set_title("Overlay", fontsize=10)
            axes[row][2].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = out_dir / f"{class_name}_features.png"
        fig.savefig(str(save_path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [viz] {class_name:30s} -> {save_path.name}")

    print(f"[viz] Feature maps saved to {out_dir}\n")


def generate_training_curves(
    history: dict, out_dir: Path
) -> None:
    """Plot and save training loss / accuracy curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss over Epochs")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy over Epochs")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "training_curves.png"
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves saved to {path}")


def generate_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    out_dir: Path,
) -> None:
    """Generate and save a confusion matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    n = len(class_names)
    matrix = np.zeros((n, n), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        matrix[true][pred] += 1

    fig, ax = plt.subplots(figsize=(max(10, n * 0.8), max(8, n * 0.7)))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)

    short_names = [c.replace("_", " ").title()[:15] for c in class_names]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short_names, fontsize=7)

    # Annotate cells
    thresh = matrix.max() / 2
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0:
                ax.text(j, i, str(matrix[i, j]),
                        ha="center", va="center", fontsize=6,
                        color="white" if matrix[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    path = out_dir / "confusion_matrix.png"
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved to {path}")


def evaluate_test_set(
    model: nn.Module,
    class_names: list[str],
    device: torch.device,
    out_dir: Path,
) -> None:
    """Evaluate the model on the hand-curated test set in data/test/."""
    test_ds = load_test_dataset(class_names, transform=test_transforms)
    if test_ds is None or len(test_ds) == 0:
        print("  [test] No test data available — skipping.")
        return

    print(f"\n  Test set: {len(test_ds)} images")
    model.eval()
    correct = 0
    results: list[str] = []

    with torch.no_grad():
        for i in range(len(test_ds)):
            img, label = test_ds[i]
            img_tensor = img.unsqueeze(0).to(device)
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = probs.max(1)
            pred_idx = pred.item()
            confidence = conf.item()
            true_name = class_names[label]
            pred_name = class_names[pred_idx]
            match = "OK" if pred_idx == label else "MISS"
            if pred_idx == label:
                correct += 1
            img_path = Path(test_ds.samples[i][0]).name
            line = (f"  {img_path:35s}  true={true_name:25s}  "
                    f"pred={pred_name:25s}  conf={confidence:.3f}  {match}")
            results.append(line)
            print(line)

    acc = correct / len(test_ds) if len(test_ds) > 0 else 0.0
    print(f"\n  Test accuracy: {correct}/{len(test_ds)} = {acc:.1%}")

    # Save results to file
    report_path = out_dir / "test_results.txt"
    with open(report_path, "w") as f:
        f.write(f"Test Evaluation Results\n{'='*70}\n")
        for line in results:
            f.write(line + "\n")
        f.write(f"\nAccuracy: {correct}/{len(test_ds)} = {acc:.1%}\n")
    print(f"  Results saved to {report_path}")


# ── training loop ────────────────────────────────────────────────────
def train(epochs: int = 60, lr: float = 1e-3, unfreeze_epoch: int = 20) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Data ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading scraped training data ...")
    scraped_summary()

    full_dataset = load_scraped_dataset(transform=train_transforms)
    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes
    print(f"  {len(full_dataset)} frames, {num_classes} classes")

    # Train/val split (80/20)
    n_total = len(full_dataset)
    n_val = max(num_classes, int(n_total * 0.2))  # at least 1 per class
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"  Split: {n_train} train / {n_val} val")

    train_loader = DataLoader(
        train_ds,
        batch_size=min(16, n_train),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(16, n_val),
        shuffle=False,
        num_workers=0,
    )

    # ── Model ────────────────────────────────────────────────────────
    print(f"\n[2/4] Building ResNet-18 (transfer learning) ...")
    model = build_model(num_classes, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # ── Train ────────────────────────────────────────────────────────
    print(f"\n[3/4] Training for {epochs} epochs ...")
    print(f"  Backbone unfreezes at epoch {unfreeze_epoch}\n")
    t0 = time.time()

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Unfreeze backbone partway through
        if epoch == unfreeze_epoch:
            print(f"  -- Unfreezing backbone at epoch {epoch} --")
            unfreeze_backbone(model)
            optimizer = optim.Adam([
                {"params": model.fc.parameters(), "lr": lr * 0.5},
                {"params": (p for n, p in model.named_parameters()
                            if "fc" not in n and p.requires_grad),
                 "lr": lr * 0.01},
            ])
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.5
            )

        # ── Training pass ────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        # ── Validation pass ──────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"loss={train_loss:.4f}  acc={train_acc:.3f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
                  f"lr={lr_now:.1e}")

    elapsed = time.time() - t0
    print(f"\n  Training completed in {elapsed:.1f}s.")
    print(f"  Best validation accuracy: {best_val_acc:.3f}")

    # Load best model for visualization
    model.load_state_dict(
        torch.load(MODEL_DIR / "best_model.pth", weights_only=True)
    )

    # ── Save final model + metadata ──────────────────────────────────
    model_path = MODEL_DIR / "boulder_classifier.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n  Model weights saved to: {model_path}")

    meta = {
        "classes": class_names,
        "num_classes": num_classes,
        "img_size": IMG_SIZE,
        "architecture": "resnet18",
        "best_val_acc": best_val_acc,
        "epochs": epochs,
    }
    meta_path = MODEL_DIR / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Class metadata saved to: {meta_path}")

    # ── [4/5] Visualizations ─────────────────────────────────────────
    print(f"\n[4/5] Generating visualizations ...")

    # Training curves
    generate_training_curves(history, VIS_DIR)

    # Confusion matrix on val set
    generate_confusion_matrix(model, val_loader, class_names, device, VIS_DIR)

    # Grad-CAM feature importance per class
    generate_feature_maps(
        model, full_dataset, class_names, device, VIS_DIR,
        samples_per_class=3,
    )

    # ── [5/5] Test-set evaluation ────────────────────────────────────
    print(f"\n[5/5] Evaluating on test set ...")
    evaluate_test_set(model, class_names, device, MODEL_DIR)

    print("\n  Done!\n")


def eval_only() -> None:
    """Load saved model and run test-set evaluation only."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_path = MODEL_DIR / "model_meta.json"
    model_path = MODEL_DIR / "best_model.pth"

    if not meta_path.exists() or not model_path.exists():
        print("[ERROR] No saved model found. Run training first.")
        return

    with open(meta_path) as f:
        meta = json.load(f)
    class_names = meta["classes"]
    num_classes = meta["num_classes"]

    print(f"\nDevice: {device}")
    print(f"Loading model ({num_classes} classes) ...")
    model = build_model(num_classes, device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    print("\nEvaluating on test set ...")
    evaluate_test_set(model, class_names, device, MODEL_DIR)
    print("\n  Done!\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the boulder classifier on scraped data."
    )
    parser.add_argument("--epochs", type=int, default=60,
                        help="Number of training epochs (default: 60).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate (default: 1e-3).")
    parser.add_argument("--unfreeze", type=int, default=20,
                        help="Epoch at which to unfreeze backbone (default: 20).")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; just evaluate the saved model on the test set.")
    args = parser.parse_args()

    if args.eval_only:
        eval_only()
    else:
        train(epochs=args.epochs, lr=args.lr, unfreeze_epoch=args.unfreeze)
