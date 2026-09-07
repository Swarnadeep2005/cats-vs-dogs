"""
Rigorous evaluation of a trained model on the held-out test set.

Why this exists as a separate script from train.py:

train.py's final model.evaluate() call only gives you a single
accuracy/loss number. That's not enough to actually understand model
behavior — a model can have 95% accuracy while being consistently
wrong on one class (e.g. if the dataset is imbalanced), and accuracy
alone would hide that. This script produces the fuller picture a
resume reviewer or interviewer would expect:

- A confusion matrix (where exactly is the model wrong?)
- Precision, recall, F1 per class (not just overall accuracy)
- ROC curve and AUC (how good is the model across all thresholds,
  not just the default 0.5 cutoff?)
- A grid of misclassified examples (what kinds of images fool it?)

All of this is computed strictly on the test set built by
build_datasets(), which was never touched during training or
hyperparameter tuning (see preprocess.py) — so these numbers are the
honest, final ones to put in a README or resume.

Usage:
    python -m src.evaluate --model model/mobilenetv2-v1.keras
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from tensorflow import keras

sys.path.append(str(Path(__file__).resolve().parents[1]))  # allow `src.` imports when run directly

from src.data.preprocess import load_config, build_datasets


def get_predictions_and_labels(model, test_ds):
    """Runs the model over the full test set and collects predictions.

    Returns (y_true, y_pred_probs, images) where y_pred_probs are the
    raw sigmoid outputs (not yet thresholded), so callers can compute
    threshold-independent metrics like ROC-AUC as well as thresholded
    ones like the confusion matrix.
    """
    y_true = []
    y_pred_probs = []
    images_for_display = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0).flatten()
        y_true.extend(labels.numpy().tolist())
        y_pred_probs.extend(preds.tolist())
        images_for_display.append(images.numpy())

    return (
        np.array(y_true),
        np.array(y_pred_probs),
        np.concatenate(images_for_display, axis=0),
    )


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Test Set")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {output_path}")
    return cm


def plot_roc_curve(y_true, y_pred_probs, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    auc = roc_auc_score(y_true, y_pred_probs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"ROC curve saved to {output_path} (AUC = {auc:.4f})")
    return auc


def plot_misclassified_examples(images, y_true, y_pred, class_names, output_path, num_examples=9):
    """Saves a grid of images the model got wrong, with true vs
    predicted labels — useful for spotting patterns in failure modes
    (e.g. blurry images, unusual poses, multiple animals in frame).
    """
    misclassified_idx = np.where(y_true != y_pred)[0]

    if len(misclassified_idx) == 0:
        print("No misclassified examples found — skipping misclassification grid.")
        return

    num_examples = min(num_examples, len(misclassified_idx))
    chosen = np.random.choice(misclassified_idx, num_examples, replace=False)

    grid_size = int(np.ceil(np.sqrt(num_examples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(3 * grid_size, 3 * grid_size))
    axes = np.array(axes).flatten()

    for ax, idx in zip(axes, chosen):
        ax.imshow(images[idx])
        ax.set_title(
            f"True: {class_names[int(y_true[idx])]}\nPred: {class_names[int(y_pred[idx])]}",
            fontsize=10,
        )
        ax.axis("off")

    for ax in axes[len(chosen):]:
        ax.axis("off")

    fig.suptitle("Misclassified Examples", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Misclassified examples grid saved to {output_path}")


def evaluate(model_path: str):
    config = load_config()
    _, _, test_ds, class_names = build_datasets(config)

    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)

    print("Running predictions on test set...")
    y_true, y_pred_probs, images = get_predictions_and_labels(model, test_ds)
    y_pred = (y_pred_probs >= 0.5).astype(int)

    # Namespace outputs by model filename so evaluating one model
    # doesn't overwrite another's artifacts (this was a real bug in
    # an earlier version — running evaluate.py twice for two
    # different models silently overwrote the first model's images).
    model_name = Path(model_path).stem  # e.g. "baseline_cnn-v1"
    output_dir = Path("logs/evaluation") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Classification Report ---")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(report)

    plot_confusion_matrix(y_true, y_pred, class_names, output_dir / "confusion_matrix.png")
    auc = plot_roc_curve(y_true, y_pred_probs, output_dir / "roc_curve.png")
    plot_misclassified_examples(
        images, y_true, y_pred, class_names, output_dir / "misclassified_examples.png"
    )

    print(f"\nAll evaluation artifacts saved to {output_dir}/")
    print(f"Summary: AUC = {auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained .keras model file, e.g. model/mobilenetv2-v1.keras",
    )
    args = parser.parse_args()
    evaluate(args.model)
