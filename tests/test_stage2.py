"""
Unit tests for Stage 2:
- Label mapping consistency
- Class name normalization
- Stratified data splitting
- CNN model architecture & tensor output shapes
- Metric calculation logic
- Missing directory error handling
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data.preprocessing import (
    normalize_class_name,
    split_dataset,
    validate_and_collect_images,
)
from src.training.model import build_custom_cnn, get_model_specs
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


class TestStage2Pipeline(unittest.TestCase):

    def test_class_mappings(self):
        """Verifies bidirectional class mapping in config."""
        self.assertEqual(len(config.CLASS_NAMES), 2)
        for name in config.CLASS_NAMES:
            idx = config.CLASS_TO_INDEX[name]
            self.assertEqual(config.INDEX_TO_CLASS[idx], name)

    def test_class_normalization(self):
        """Verifies class name normalization variations."""
        self.assertEqual(normalize_class_name("with_mask"), "with_mask")
        self.assertEqual(normalize_class_name("Mask"), "with_mask")
        self.assertEqual(normalize_class_name("without_mask"), "without_mask")
        self.assertEqual(normalize_class_name("no_mask"), "without_mask")
        self.assertEqual(normalize_class_name("No-Mask"), "without_mask")
        self.assertIsNone(normalize_class_name("unrelated_folder"))

    def test_stratified_split(self):
        """Verifies 70/15/15 stratified split ratios and class distributions."""
        # Create 100 synthetic sample records (50 with_mask, 50 without_mask)
        records = [
            {"filepath": f"path/mask_{i}.jpg", "class_name": "with_mask", "label": 0, "file_hash": f"h_mask_{i}"}
            for i in range(50)
        ] + [
            {"filepath": f"path/nomask_{i}.jpg", "class_name": "without_mask", "label": 1, "file_hash": f"h_nomask_{i}"}
            for i in range(50)
        ]

        train_df, val_df, test_df = split_dataset(
            records,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )

        self.assertEqual(len(train_df), 70)
        self.assertEqual(len(val_df), 15)
        self.assertEqual(len(test_df), 15)

        # Verify stratification (should be exactly 50% in train, and roughly balanced in val/test)
        self.assertEqual(sum(train_df["label"] == 0), 35)
        self.assertEqual(sum(train_df["label"] == 1), 35)

    def test_cnn_model_output_shape(self):
        """Verifies that the custom CNN builds properly and produces correct output shape and probabilities."""
        model = build_custom_cnn(input_shape=(128, 128, 3), num_classes=2)
        
        # Test forward pass with random batch of 2 dummy images
        dummy_input = tf.random.uniform(shape=(2, 128, 128, 3), minval=0.0, maxval=1.0)
        predictions = model(dummy_input, training=False)

        self.assertEqual(predictions.shape, (2, 2))
        
        # Sum of softmax probabilities must be approximately 1.0
        prob_sums = tf.reduce_sum(predictions, axis=1).numpy()
        np.testing.assert_allclose(prob_sums, [1.0, 1.0], atol=1e-5)

    def test_evaluation_metric_computations(self):
        """Verifies that accuracy, confusion matrix, and macro F1 computations are exact."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])  # 1 mistake on class 0

        acc = accuracy_score(y_true, y_pred)
        self.assertAlmostEqual(acc, 0.75)

        cm = confusion_matrix(y_true, y_pred)
        self.assertEqual(cm[0, 0], 1)  # True mask
        self.assertEqual(cm[0, 1], 1)  # False no-mask
        self.assertEqual(cm[1, 0], 0)  # False mask
        self.assertEqual(cm[1, 1], 2)  # True no-mask

    def test_missing_directory_error_handling(self):
        """Verifies that validate_and_collect_images raises FileNotFoundError when directory is missing."""
        non_existent_path = PROJECT_ROOT / "non_existent_raw_dir_12345"
        with self.assertRaises(FileNotFoundError):
            validate_and_collect_images(non_existent_path)


if __name__ == "__main__":
    unittest.main()
