"""
Unit and Integration tests for Stage 3:
- FaceMaskDetector initialization and model loading.
- Crop preprocessing and CNN inference.
- Handling of images with no faces.
- Flask endpoints (/api/status, /api/predict-image, /api/predict-frame).
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.detection.detector import FaceMaskDetector
from app.app import app


class TestStage3DetectionAndApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initializes detector and Flask test client."""
        cls.detector = FaceMaskDetector()
        cls.client = app.test_client()

    def test_detector_model_loaded(self):
        """Verifies that the trained CNN is loaded into the detector."""
        self.assertIsNotNone(self.detector.model)
        self.assertIn("with_mask", self.detector.idx_to_class.values())
        self.assertIn("without_mask", self.detector.idx_to_class.values())

    def test_predict_face_crop(self):
        """Tests prediction on a dummy 128x128 BGR face crop."""
        dummy_face = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        label, confidence, class_id = self.detector.predict_face_crop(dummy_face)

        self.assertIn(label, ["MASK", "NO MASK"])
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertIn(class_id, [0, 1])

    def test_no_face_detection_handling(self):
        """Verifies that an image with no faces (e.g. solid black image) returns 0 detections cleanly."""
        black_image = np.zeros((300, 300, 3), dtype=np.uint8)
        result = self.detector.process_image(black_image)

        self.assertEqual(result["faces_detected"], 0)
        self.assertEqual(len(result["detections"]), 0)
        self.assertEqual(result["mask_count"], 0)
        self.assertEqual(result["no_mask_count"], 0)
        self.assertEqual(result["annotated_image"].shape, (300, 300, 3))

    def test_flask_status_endpoint(self):
        """Tests GET /api/status returns 200 and expected metadata."""
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["model"], "custom_face_mask_cnn")
        self.assertEqual(data["device"], "CPU")

    def test_flask_index_page(self):
        """Tests GET / returns index.html template with 200 OK."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"FaceShield", response.data)


if __name__ == "__main__":
    unittest.main()
