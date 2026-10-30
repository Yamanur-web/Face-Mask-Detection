"""
Custom Lightweight Convolutional Neural Network (CNN) for Face Mask Classification.

Architecture:
- Input: 128 x 128 x 3 RGB images normalized to [0, 1].
- 4 Convolutional feature extraction blocks with ReLU and Batch Normalization.
- MaxPooling layers to downsample spatial dimensions.
- GlobalAveragePooling2D to prevent parameter explosion.
- Dense layer with Dropout (0.5) to regularize and prevent overfitting.
- 2-unit Softmax output layer: [0: with_mask, 1: without_mask].
- Lightweight footprint (~250,000 parameters) engineered for CPU-only training on Intel Core i5.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict

import tensorflow as tf

# Ensure src modules are resolvable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config


def build_custom_cnn(
    input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
    num_classes: int = config.NUM_CLASSES,
    learning_rate: float = config.INITIAL_LEARNING_RATE,
) -> tf.keras.Model:
    """
    Builds and compiles the custom CNN classifier.
    Does not use any pretrained models or transfer learning.
    """
    inputs = tf.keras.Input(shape=input_shape, name="face_input_rgb")

    # Feature Extraction Block 1: 128x128 -> 64x64
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_32")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # Feature Extraction Block 2: 64x64 -> 32x32
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_64")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # Feature Extraction Block 3: 32x32 -> 16x16
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_128")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    # Feature Extraction Block 4: 16x16 -> 8x8
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv4_128")(x)
    x = tf.keras.layers.BatchNormalization(name="bn4")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)

    # Classification Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout_regularization")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="mask_probabilities")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="custom_face_mask_cnn")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_model_summary(model: tf.keras.Model) -> str:
    """Returns the string summary of the model."""
    summary_lines = []
    model.summary(print_fn=lambda line: summary_lines.append(line))
    return "\n".join(summary_lines)


def get_model_specs() -> Dict:
    """Returns metadata describing architecture and class mapping."""
    return {
        "model_name": "custom_face_mask_cnn",
        "input_shape": list(config.INPUT_SHAPE),
        "num_classes": config.NUM_CLASSES,
        "class_mapping": config.CLASS_TO_INDEX,
        "index_to_class": config.INDEX_TO_CLASS,
        "normalization": "pixel / 255.0 to [0, 1]",
    }


if __name__ == "__main__":
    print("[MODEL] Building custom Face Mask CNN architecture...")
    cnn_model = build_custom_cnn()
    print(get_model_summary(cnn_model))
    print(f"\n[MODEL] Total parameters: {cnn_model.count_params():,}")
    print(f"[MODEL] Class index mapping: {config.CLASS_TO_INDEX}")
