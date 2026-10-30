"""
Model Evaluation Pipeline for Face Mask Detection.

Evaluates the trained custom CNN strictly on the isolated test set:
- Computes Overall Accuracy, Precision, Recall, and Macro F1-score.
- Computes Per-Class Precision, Recall, and F1-score.
- Computes and plots the Confusion Matrix.
- Saves results to reports/evaluation_metrics.json and reports/figures/confusion_matrix.png.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# Ensure src modules are resolvable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data.preprocessing import create_tf_dataset


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: Path,
):
    """Plots and saves an annotated confusion matrix heatmap."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Test Set Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # Annotate counts inside the squares
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
                fontweight="bold",
            )

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[REPORTS] Confusion matrix plot saved to: {save_path}")


def evaluate_test_set(
    model_path: Path = config.SAVED_MODEL_PATH,
    test_manifest_path: Path = config.TEST_MANIFEST_PATH,
    batch_size: int = config.BATCH_SIZE,
) -> Dict:
    """Evaluates the saved model against the isolated test dataset."""
    # 1. Validate files exist
    if not model_path.exists():
        raise FileNotFoundError(
            f"[ERROR] Trained model file not found at: {model_path}\n"
            "Please train the model first by running:\n"
            "    python src/training/train.py\n"
        )

    if not test_manifest_path.exists():
        raise FileNotFoundError(
            f"[ERROR] Test manifest not found at: {test_manifest_path}\n"
            "Please run the preprocessing pipeline first:\n"
            "    python src/data/preprocessing.py\n"
        )

    print(f"[EVAL] Loading trained model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"[EVAL] Loading test manifest from: {test_manifest_path}")
    test_df = pd.read_csv(test_manifest_path)
    y_true = test_df["label"].values.astype(int)

    test_ds = create_tf_dataset(test_df, is_training=False, batch_size=batch_size)

    # 2. Predict on test set
    print(f"[EVAL] Running inference across {len(test_df)} test samples...")
    raw_predictions = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(raw_predictions, axis=1)

    # 3. Compute Metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)
    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=config.CLASS_NAMES,
        digits=4,
        output_dict=True,
    )
    clf_report_text = classification_report(
        y_true,
        y_pred,
        target_names=config.CLASS_NAMES,
        digits=4,
    )

    # 4. Display Results
    print("\n" + "=" * 65)
    print("           FACE MASK DETECTION — TEST EVALUATION REPORT")
    print("=" * 65)
    print(f"Total Test Samples : {len(test_df)}")
    print(f"Overall Accuracy   : {acc * 100:.2f}%")
    print(f"Macro Precision    : {macro_p:.4f}")
    print(f"Macro Recall       : {macro_r:.4f}")
    print(f"Macro F1-Score     : {macro_f1:.4f}")
    print("-" * 65)
    print("\n[PER-CLASS BREAKDOWN]")
    print(clf_report_text)
    print("-" * 65)
    print("\n[CONFUSION MATRIX]")
    print(f"               Predicted {config.CLASS_NAMES[0]} | Predicted {config.CLASS_NAMES[1]}")
    print(f"True {config.CLASS_NAMES[0]:<10}: {cm[0][0]:<15} | {cm[0][1]}")
    print(f"True {config.CLASS_NAMES[1]:<10}: {cm[1][0]:<15} | {cm[1][1]}")
    print("=" * 65 + "\n")

    # 5. Save Artifacts
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results_payload = {
        "test_samples": len(test_df),
        "overall_accuracy": acc,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1_score": float(macro_f1),
        "per_class_metrics": {
            cls_name: {
                "precision": float(per_class_p[idx]),
                "recall": float(per_class_r[idx]),
                "f1_score": float(per_class_f1[idx]),
                "support": int(per_class_support[idx]),
            }
            for idx, cls_name in enumerate(config.CLASS_NAMES)
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report,
    }

    with open(config.EVALUATION_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=4)
    print(f"[REPORTS] Evaluation metrics JSON saved to: {config.EVALUATION_METRICS_PATH}")

    plot_confusion_matrix(cm, config.CLASS_NAMES, config.CONFUSION_MATRIX_PLOT_PATH)

    return results_payload


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained CNN on isolated test set.")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size for test evaluation")
    args = parser.parse_args()

    try:
        evaluate_test_set(batch_size=args.batch_size)
    except Exception as e:
        print(f"\n{e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
