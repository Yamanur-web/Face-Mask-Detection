"""
Model Training Pipeline for Face Mask Detection.

Features:
- Reproducibility via seed setting (Python, NumPy, TensorFlow).
- Batch-by-batch data loading to prevent laptop RAM overload.
- Callbacks: EarlyStopping (restore best weights), ReduceLROnPlateau, ModelCheckpoint.
- Saves:
  1. Best model weights (.keras)
  2. Class indices mapping (.json)
  3. Model metadata (.json)
  4. Training curves plot (accuracy & loss)
"""

import sys
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Ensure src modules are resolvable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data.preprocessing import create_tf_dataset
from src.training.model import build_custom_cnn, get_model_specs


def set_seed(seed: int = config.RANDOM_SEED):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def plot_training_curves(history: tf.keras.callbacks.History, save_path: Path):
    """Generates and saves accuracy and loss progression plots."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs_range = range(1, len(history.history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss Curves
    ax1.plot(epochs_range, history.history["loss"], "b-o", label="Training Loss", linewidth=2)
    ax1.plot(epochs_range, history.history["val_loss"], "r--s", label="Validation Loss", linewidth=2)
    ax1.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")

    # Accuracy Curves
    ax2.plot(epochs_range, history.history["accuracy"], "b-o", label="Training Accuracy", linewidth=2)
    ax2.plot(epochs_range, history.history["val_accuracy"], "g--s", label="Validation Accuracy", linewidth=2)
    ax2.set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[REPORTS] Training curves saved to: {save_path}")


def train_pipeline(
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.INITIAL_LEARNING_RATE,
) -> Dict:
    """Orchestrates model building, data loading, training, and artifact saving."""
    set_seed(config.RANDOM_SEED)

    # 1. Validate manifests exist
    if not config.TRAIN_MANIFEST_PATH.exists() or not config.VAL_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"[ERROR] Processed data manifests not found!\n"
            f"Expected: {config.TRAIN_MANIFEST_PATH}\n"
            "Please run the preprocessing pipeline first:\n"
            "    python src/data/preprocessing.py\n"
        )

    print("[TRAIN] Loading dataset manifests...")
    train_df = pd.read_csv(config.TRAIN_MANIFEST_PATH)
    val_df = pd.read_csv(config.VAL_MANIFEST_PATH)

    print(f"[TRAIN] Training samples   : {len(train_df)}")
    print(f"[TRAIN] Validation samples : {len(val_df)}")

    # 2. Build tf.data streaming datasets (RAM friendly)
    train_ds = create_tf_dataset(train_df, is_training=True, batch_size=batch_size)
    val_ds = create_tf_dataset(val_df, is_training=False, batch_size=batch_size)

    # 3. Build model
    print("\n[TRAIN] Building custom CNN architecture...")
    model = build_custom_cnn(
        input_shape=config.INPUT_SHAPE,
        num_classes=config.NUM_CLASSES,
        learning_rate=learning_rate,
    )

    # 4. Ensure directories exist
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 5. Configure Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.SAVED_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # 6. Train the model
    print(f"\n[TRAIN] Starting CPU training: {epochs} epochs max, batch size {batch_size}...")
    start_time = datetime.now()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed_time = str(datetime.now() - start_time)
    print(f"\n[TRAIN] Training completed in: {elapsed_time}")

    # 7. Save class indices JSON
    with open(config.CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(config.CLASS_TO_INDEX, f, indent=4)
    print(f"[ARTIFACTS] Class indices saved to: {config.CLASS_INDICES_PATH}")

    # 8. Save metadata
    metadata = {
        "model_architecture": "custom_face_mask_cnn",
        "input_shape": list(config.INPUT_SHAPE),
        "total_parameters": model.count_params(),
        "trainable_parameters": int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)),
        "class_mapping": config.CLASS_TO_INDEX,
        "index_to_class": config.INDEX_TO_CLASS,
        "normalization": "pixel / 255.0 to [0, 1]",
        "training_samples": len(train_df),
        "validation_samples": len(val_df),
        "batch_size": batch_size,
        "epochs_trained": len(history.history["loss"]),
        "training_duration": elapsed_time,
        "best_val_loss": float(min(history.history["val_loss"])),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "model_file": str(config.SAVED_MODEL_PATH.name),
    }

    with open(config.MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[ARTIFACTS] Model metadata saved to: {config.MODEL_METADATA_PATH}")

    # 9. Plot and save training history curves
    plot_training_curves(history, config.TRAINING_HISTORY_PLOT_PATH)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train custom CNN for Face Mask Detection.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.INITIAL_LEARNING_RATE, help="Initial learning rate")
    args = parser.parse_args()

    try:
        train_pipeline(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
    except Exception as e:
        print(f"\n{e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
