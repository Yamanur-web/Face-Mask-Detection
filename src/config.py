"""
Configuration parameters for Face Mask Detection project.
Centralizes paths, hyperparameters, image dimensions, and class definitions.
"""

from pathlib import Path

# Base Paths (Face_Mask_Detection project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Image & Preprocessing Specifications
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Class Label Definitions
CLASS_NAMES = ["with_mask", "without_mask"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Training Hyperparameters (Lightweight CPU-friendly)
BATCH_SIZE = 32
EPOCHS = 20
INITIAL_LEARNING_RATE = 0.001
RANDOM_SEED = 42

# Data Split Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Manifest and Processed Data Paths
TRAIN_MANIFEST_PATH = PROCESSED_DATA_DIR / "train_manifest.csv"
VAL_MANIFEST_PATH = PROCESSED_DATA_DIR / "val_manifest.csv"
TEST_MANIFEST_PATH = PROCESSED_DATA_DIR / "test_manifest.csv"
SPLIT_SUMMARY_PATH = PROCESSED_DATA_DIR / "split_summary.json"

# Model Artifacts
SAVED_MODEL_FILENAME = "best_mask_detector_cnn.keras"
SAVED_MODEL_PATH = MODELS_DIR / SAVED_MODEL_FILENAME
CLASS_INDICES_FILENAME = "class_indices.json"
CLASS_INDICES_PATH = MODELS_DIR / CLASS_INDICES_FILENAME
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Evaluation and Reporting Paths
TRAINING_HISTORY_PLOT_PATH = FIGURES_DIR / "training_curves.png"
CONFUSION_MATRIX_PLOT_PATH = FIGURES_DIR / "confusion_matrix.png"
EVALUATION_METRICS_PATH = REPORTS_DIR / "evaluation_metrics.json"

# OpenCV Face Detection Configuration
HAAR_CASCADE_NAME = "haarcascade_frontalface_default.xml"
MIN_FACE_SIZE = (60, 60)
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
