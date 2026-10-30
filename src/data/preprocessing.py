"""
Dataset validation, cleaning, deduplication, stratified splitting, and tf.data pipeline.

Ensures:
1. Exact class detection and consistent label mapping (Mask -> 0, No Mask -> 1).
2. Corrupted and zero-byte file detection.
3. Hash-based deduplication to prevent data leakage.
4. Stratified 70/15/15 train/val/test splits saved to manifest files.
5. Training-only data augmentation (horizontal flips, slight rotation & zoom).
6. RAM-efficient batch streaming via tf.data.Dataset with prefetching.
"""

import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Ensure src modules are resolvable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Standard canonical class mappings
CANONICAL_CLASS_MAP = {
    "with_mask": "with_mask",
    "with-mask": "with_mask",
    "mask": "with_mask",
    "masked": "with_mask",
    "without_mask": "without_mask",
    "without-mask": "without_mask",
    "no_mask": "without_mask",
    "no-mask": "without_mask",
    "nomask": "without_mask",
    "unmasked": "without_mask",
}


def normalize_class_name(name: str) -> Optional[str]:
    """Normalizes class folder name variations to canonical 'with_mask' or 'without_mask'."""
    clean = name.strip().lower().replace(" ", "_")
    return CANONICAL_CLASS_MAP.get(clean)


def compute_file_hash(filepath: Path) -> str:
    """Computes SHA-256 hash of file content to detect duplicate images."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_and_collect_images(raw_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """
    Validates images in raw_dir.
    Returns:
        valid_records: list of dicts with filepath, canonical_class, label_idx, file_hash.
        corrupted_records: list of dicts with filepath, reason.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"[ERROR] Raw dataset directory does not exist: {raw_dir}\n"
            f"Please download the dataset and extract it into '{raw_dir}'."
        )

    subdirs = [d for d in raw_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    # Support single-nested root folder if archive extracted into an outer folder
    if len(subdirs) == 1:
        nested = [d for d in subdirs[0].iterdir() if d.is_dir() and not d.name.startswith(".")]
        if len(nested) >= 2:
            print(f"[INFO] Drilling down into nested wrapper folder: '{subdirs[0].name}'")
            subdirs = nested

    if not subdirs:
        raise FileNotFoundError(
            f"[ERROR] No class directories found in: {raw_dir}\n"
            "Expected folders such as 'with_mask' and 'without_mask'.\n"
            "Please extract the dataset into dataset/raw/ first."
        )

    class_dirs = {}
    for d in subdirs:
        canonical = normalize_class_name(d.name)
        if canonical is not None:
            class_dirs[canonical] = d
        else:
            print(f"[WARN] Ignoring directory with unrecognized class name: '{d.name}'")

    missing_classes = set(config.CLASS_NAMES) - set(class_dirs.keys())
    if missing_classes:
        raise ValueError(
            f"[ERROR] Missing required classes: {missing_classes}. "
            f"Found classes: {list(class_dirs.keys())}. Required: {config.CLASS_NAMES}"
        )

    valid_records = []
    corrupted_records = []
    seen_hashes = {}

    for canonical_name, class_dir in class_dirs.items():
        label_idx = config.CLASS_TO_INDEX[canonical_name]
        files = list(class_dir.rglob("*"))

        for file_path in files:
            if file_path.is_dir() or file_path.name.startswith("."):
                continue

            if file_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            if file_path.stat().st_size == 0:
                corrupted_records.append({"path": str(file_path), "reason": "Zero-byte file"})
                continue

            try:
                # Verify image readability and structure
                with Image.open(file_path) as img:
                    img.verify()

                # Deduplication check
                file_hash = compute_file_hash(file_path)
                if file_hash in seen_hashes:
                    # Skip duplicate to prevent data leakage across splits
                    continue

                seen_hashes[file_hash] = file_path
                # Store path relative to PROJECT_ROOT for portability
                rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()
                valid_records.append({
                    "filepath": rel_path,
                    "class_name": canonical_name,
                    "label": label_idx,
                    "file_hash": file_hash,
                })
            except (UnidentifiedImageError, OSError, Exception) as exc:
                corrupted_records.append({"path": str(file_path), "reason": str(exc)})

    return valid_records, corrupted_records


def split_dataset(
    records: List[Dict],
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    random_seed: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits records into stratified train, validation, and test DataFrames.
    """
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("[ERROR] No valid images found to split.")

    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["label"],
        random_state=random_seed,
    )

    # Second split: Val vs Test
    relative_test_ratio = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df["label"],
        random_state=random_seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_manifests(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path = config.PROCESSED_DATA_DIR,
) -> Dict:
    """Saves manifest CSVs and summary JSON to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_manifest.csv"
    val_path = output_dir / "val_manifest.csv"
    test_path = output_dir / "test_manifest.csv"
    summary_path = output_dir / "split_summary.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "total_images": len(train_df) + len(val_df) + len(test_df),
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "train_class_counts": train_df["class_name"].value_counts().to_dict(),
        "val_class_counts": val_df["class_name"].value_counts().to_dict(),
        "test_class_counts": test_df["class_name"].value_counts().to_dict(),
        "class_mapping": config.CLASS_TO_INDEX,
        "random_seed": config.RANDOM_SEED,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return summary


def get_data_augmentation_layer() -> tf.keras.Sequential:
    """Builds a lightweight data augmentation pipeline for training only."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08, fill_mode="nearest"),
            tf.keras.layers.RandomZoom(0.08, fill_mode="nearest"),
        ],
        name="training_augmentation",
    )


def load_and_preprocess_image(filepath_tensor: tf.Tensor, label_tensor: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Loads an image file, decodes it as RGB, resizes to 128x128, and normalizes to [0, 1].
    """
    img_bytes = tf.io.read_file(filepath_tensor)
    # decode_image handles JPEG, PNG, BMP, etc.
    img = tf.io.decode_image(img_bytes, channels=config.IMG_CHANNELS, expand_animations=False)
    img = tf.image.resize(img, [config.IMG_HEIGHT, config.IMG_WIDTH])
    # Normalize to [0, 1]
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape(config.INPUT_SHAPE)
    return img, label_tensor


def create_tf_dataset(
    manifest_df: pd.DataFrame,
    is_training: bool = False,
    batch_size: int = config.BATCH_SIZE,
    shuffle_buffer: int = 1000,
) -> tf.data.Dataset:
    """
    Creates an efficient tf.data.Dataset from a manifest DataFrame.
    Images are loaded batch-by-batch on the fly to avoid overloading laptop RAM.
    """
    # Prepend PROJECT_ROOT to relative paths
    absolute_paths = [str(PROJECT_ROOT / p) for p in manifest_df["filepath"]]
    labels = manifest_df["label"].values.astype(np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((absolute_paths, labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(manifest_df), shuffle_buffer), seed=config.RANDOM_SEED)

    # Parallel mapping with AUTOTUNE (deterministic when not training to preserve test order)
    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not is_training,
    )

    # Apply data augmentation ONLY if is_training is True
    if is_training:
        aug_layer = get_data_augmentation_layer()
        dataset = dataset.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def run_preprocessing_pipeline() -> Dict:
    """End-to-end execution of validation, deduplication, splitting, and manifest generation."""
    print(f"[DATA] Scanning raw dataset in: {config.RAW_DATA_DIR}")
    valid_records, corrupted = validate_and_collect_images(config.RAW_DATA_DIR)

    if corrupted:
        print(f"[WARN] Found {len(corrupted)} corrupted/unreadable images:")
        for rec in corrupted[:5]:
            print(f"  - {rec['path']} ({rec['reason']})")
        if len(corrupted) > 5:
            print(f"  ... and {len(corrupted) - 5} more.")

    print(f"[DATA] Valid unique images collected: {len(valid_records)}")
    train_df, val_df, test_df = split_dataset(valid_records)

    print(f"[SPLIT] Training Set   : {len(train_df)} ({len(train_df)/len(valid_records)*100:.1f}%)")
    print(f"[SPLIT] Validation Set : {len(val_df)} ({len(val_df)/len(valid_records)*100:.1f}%)")
    print(f"[SPLIT] Test Set       : {len(test_df)} ({len(test_df)/len(valid_records)*100:.1f}%)")

    summary = save_manifests(train_df, val_df, test_df)
    print(f"[SUCCESS] Manifests saved to: {config.PROCESSED_DATA_DIR}")
    return summary


if __name__ == "__main__":
    try:
        run_preprocessing_pipeline()
    except Exception as e:
        print(f"\n{e}\n")
        sys.exit(1)
