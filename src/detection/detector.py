"""
End-to-End Face Mask Detection Pipeline.

Combines:
1. OpenCV Haar Cascade Classifier for rapid face detection and bounding box localization.
2. Custom CNN model inference for Mask vs. No Mask classification with confidence score.
3. High-quality annotation utilities with color-coded bounding boxes and label banners.
"""

import sys
import json
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import tensorflow as tf

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config


class FaceMaskDetector:
    """
    Two-stage Face Mask Detection pipeline:
    Stage A: Detect face ROI using OpenCV Haar Cascade.
    Stage B: Classify cropped face ROI using custom trained CNN.
    """

    def __init__(
        self,
        model_path: Path = config.SAVED_MODEL_PATH,
        class_indices_path: Path = config.CLASS_INDICES_PATH,
    ):
        self.model_path = Path(model_path)
        self.class_indices_path = Path(class_indices_path)

        # 1. Load OpenCV Haar Cascade Face Detector
        cascade_path = cv2.data.haarcascades + config.HAAR_CASCADE_NAME

        # Resolve CascadeClassifier dynamically from cv2 or cv2.objdetect
        cascade_cls = getattr(cv2, "CascadeClassifier", None)
        if cascade_cls is None and hasattr(cv2, "objdetect"):
            cascade_cls = getattr(cv2.objdetect, "CascadeClassifier", None)

        if cascade_cls is None:
            raise AttributeError(
                "[ERROR] OpenCV is missing 'CascadeClassifier'. "
                "OpenCV 5.0.0 removed the legacy CascadeClassifier API. "
                "Fix it by installing the stable 4.x version:\n"
                "    pip install \"opencv-python<5.0.0\"\n"
            )

        self.face_cascade = cascade_cls(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"[ERROR] Failed to load OpenCV Haar Cascade from: {cascade_path}")

        # 2. Load trained CNN model
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"[ERROR] Trained CNN model file not found at: {self.model_path}\n"
                "Please run training first: python src/training/train.py"
            )
        print(f"[DETECTOR] Loading trained CNN model from: {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)

        # 3. Load Class Mapping
        if self.class_indices_path.exists():
            with open(self.class_indices_path, "r", encoding="utf-8") as f:
                class_to_idx = json.load(f)
                self.idx_to_class = {int(v): k for k, v in class_to_idx.items()}
        else:
            self.idx_to_class = config.INDEX_TO_CLASS

        # Styling colors (BGR format for OpenCV)
        # Green for Mask, Red for No Mask
        self.COLOR_MASK = (46, 204, 113)      # Emerald Green
        self.COLOR_NO_MASK = (52, 73, 235)    # Vibrant Red / Rose

    def detect_faces(self, image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detects bounding boxes (x, y, w, h) of faces in a BGR image."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Equalize histogram for contrast enhancement
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=config.SCALE_FACTOR,
            minNeighbors=config.MIN_NEIGHBORS,
            minSize=config.MIN_FACE_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def predict_face_crop(self, face_bgr: np.ndarray) -> Tuple[str, float, int]:
        """
        Preprocesses a cropped face region and predicts Mask vs. No Mask.
        Returns: (display_label, confidence, class_id)
        """
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        # Resize to CNN input dimensions (128x128)
        face_resized = cv2.resize(face_rgb, config.IMG_SIZE, interpolation=cv2.INTER_AREA)
        # Normalize to [0, 1]
        face_normalized = face_resized.astype(np.float32) / 255.0
        # Add batch dimension: shape (1, 128, 128, 3)
        input_tensor = np.expand_dims(face_normalized, axis=0)

        # Predict probabilities
        probabilities = self.model.predict(input_tensor, verbose=0)[0]
        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])

        canonical_name = self.idx_to_class.get(class_id, "unknown")
        display_label = "MASK" if canonical_name == "with_mask" else "NO MASK"

        return display_label, confidence, class_id

    def annotate_frame(
        self,
        image_bgr: np.ndarray,
        faces: List[Tuple[int, int, int, int]],
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Predicts mask status for each face, draws boxes and text banners on the image.
        """
        annotated = image_bgr.copy()
        detections = []

        for (x, y, w, h) in faces:
            # Crop face region with boundary checks
            x_min = max(0, x)
            y_min = max(0, y)
            x_max = min(annotated.shape[1], x + w)
            y_max = min(annotated.shape[0], y + h)

            face_roi = annotated[y_min:y_max, x_min:x_max]
            if face_roi.size == 0 or face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
                continue

            label, confidence, class_id = self.predict_face_crop(face_roi)
            color = self.COLOR_MASK if label == "MASK" else self.COLOR_NO_MASK
            conf_percent = f"{confidence * 100:.1f}%"
            text_str = f"{label} ({conf_percent})"

            # Draw outer bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Draw top label banner
            (text_w, text_h), baseline = cv2.getTextSize(
                text_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            banner_y1 = max(0, y - text_h - 10)
            banner_y2 = y
            cv2.rectangle(
                annotated,
                (x, banner_y1),
                (x + text_w + 10, banner_y2),
                color,
                -1,  # Filled background
            )

            # Draw label text
            cv2.putText(
                annotated,
                text_str,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            detections.append({
                "box": [x, y, w, h],
                "label": label,
                "confidence": round(confidence, 4),
                "confidence_percent": conf_percent,
                "class_id": class_id,
            })

        return annotated, detections

    def process_image(self, image_bgr: np.ndarray) -> Dict:
        """
        Full inference on a single BGR image.
        Returns dictionary containing annotated image and detection metadata.
        """
        faces = self.detect_faces(image_bgr)
        annotated, detections = self.annotate_frame(image_bgr, faces)

        mask_count = sum(1 for d in detections if d["label"] == "MASK")
        no_mask_count = sum(1 for d in detections if d["label"] == "NO MASK")

        return {
            "faces_detected": len(detections),
            "mask_count": mask_count,
            "no_mask_count": no_mask_count,
            "detections": detections,
            "annotated_image": annotated,
        }

    def process_image_bytes(self, image_bytes: bytes) -> Dict:
        """Processes raw image bytes (e.g., from file upload)."""
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("[ERROR] Could not decode uploaded image bytes.")
        return self.process_image(image_bgr)

    def process_base64_frame(self, base64_data: str) -> Dict:
        """Processes base64 encoded frame from client-side browser webcam."""
        if "," in base64_data:
            base64_data = base64_data.split(",", 1)[1]
        image_bytes = base64.b64decode(base64_data)
        result = self.process_image_bytes(image_bytes)

        # Convert annotated BGR back to base64 JPEG for browser rendering
        _, buffer = cv2.imencode(".jpg", result["annotated_image"], [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "faces_detected": result["faces_detected"],
            "mask_count": result["mask_count"],
            "no_mask_count": result["no_mask_count"],
            "detections": result["detections"],
            "annotated_b64": f"data:image/jpeg;base64,{annotated_b64}",
        }
