# Boulder Finder — Image Classifier

A scikit-learn image classifier that identifies **rock climbing boulders** in photos.

Uses HOG (Histogram of Oriented Gradients) + colour histogram features with an SVM classifier.

## Project Structure

```
boulder_finder/
├── data/
│   ├── train/
│   │   ├── boulder/        ← training images of boulders
│   │   └── not_boulder/    ← training images of non-boulders
│   └── test/
│       ├── boulder/
│       └── not_boulder/
├── models/                  ← saved model + confusion matrix
├── src/
│   ├── prepare_data.py      ← data loading & preparation
│   ├── train.py             ← feature extraction & model training
│   └── predict.py           ← classify new images
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Workflow

### 1. Add images

Place your images into the class folders:

- `data/train/boulder/` — photos containing climbing boulders
- `data/train/not_boulder/` — photos of other things (landscapes, walls, trees, etc.)
- Repeat for `data/test/` to have a held-out evaluation set

**Tip:** Aim for at least 50–100 images per class for decent results. More is better.

### 2. Verify your dataset

```bash
python src/prepare_data.py
```

This creates any missing directories and prints a count of images per class.

### 3. Train the model

```bash
# Default: SVM (best for smaller datasets)
python src/train.py

# Or pick a different model
python src/train.py --model rf    # Random Forest
python src/train.py --model gb    # Gradient Boosting
```

The trained model is saved to `models/boulder_classifier.joblib`. If test images are present, a confusion matrix is also saved.

### 4. Classify new images

```bash
# Single image
python src/predict.py path/to/photo.jpg

# Folder of images
python src/predict.py path/to/folder/
```

## Models Available

| Flag    | Model              | Notes                                    |
|---------|--------------------|------------------------------------------|
| `svm`   | SVM (RBF kernel)   | Default. Best accuracy on small datasets |
| `rf`    | Random Forest      | Fast, handles noisy data well            |
| `gb`    | Gradient Boosting  | Often top accuracy, slower to train      |
