"""
Flask Web Application Backend for Face Mask Detection.

Features:
- Image Upload Mode: Detects multiple faces, annotates bounding boxes, reports confidence.
- Live Webcam Mode: Processes video frames in real time from browser camera.
- Clear handling of cases where no face is detected.
- RESTful JSON endpoints with base64 image streaming.
"""

import sys
import io
import time
import base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response

# Ensure project root is in sys.path
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src import config
from src.detection.detector import FaceMaskDetector

# Initialize Flask application with explicit template and static paths
app = Flask(
    __name__,
    template_folder=str(APP_DIR / "templates"),
    static_folder=str(APP_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB maximum upload limit

# Global detector instance (lazy initialized)
_detector = None


def get_detector() -> FaceMaskDetector:
    """Singleton getter for the FaceMaskDetector model."""
    global _detector
    if _detector is None:
        print("[APP] Initializing FaceMaskDetector pipeline...")
        _detector = FaceMaskDetector()
    return _detector


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}


def is_allowed_file(filename: str) -> bool:
    """Checks if uploaded file has a valid image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Serves the main web dashboard."""
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    """Returns application, model, and system status metadata."""
    try:
        det = get_detector()
        return jsonify({
            "status": "ready",
            "model": "custom_face_mask_cnn",
            "model_path": str(config.SAVED_MODEL_PATH.name),
            "classes": config.CLASS_NAMES,
            "input_resolution": f"{config.IMG_WIDTH}x{config.IMG_HEIGHT}",
            "device": "CPU",
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict-image", methods=["POST"])
def predict_image():
    """
    Handles single image uploads from the UI.
    Returns detection results, counts, and base64 annotated image.
    """
    start_time = time.time()
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided in request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not is_allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        }), 400

    try:
        detector = get_detector()
        image_bytes = file.read()
        result = detector.process_image_bytes(image_bytes)

        # Encode annotated image to JPEG base64
        _, buffer = cv2.imencode(".jpg", result["annotated_image"], [cv2.IMWRITE_JPEG_QUALITY, 90])
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        latency_ms = round((time.time() - start_time) * 1000, 1)

        return jsonify({
            "success": True,
            "faces_detected": result["faces_detected"],
            "mask_count": result["mask_count"],
            "no_mask_count": result["no_mask_count"],
            "detections": result["detections"],
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
            "latency_ms": latency_ms,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predict-frame", methods=["POST"])
def predict_frame():
    """
    Handles real-time webcam frames sent from the browser Canvas.
    Processes frame and returns annotated image base64 stream.
    """
    start_time = time.time()
    data = request.get_json(silent=True)
    if not data or "frame" not in data:
        return jsonify({"success": False, "error": "No frame data provided."}), 400

    try:
        detector = get_detector()
        frame_result = detector.process_base64_frame(data["frame"])
        latency_ms = round((time.time() - start_time) * 1000, 1)

        return jsonify({
            "success": True,
            "faces_detected": frame_result["faces_detected"],
            "mask_count": frame_result["mask_count"],
            "no_mask_count": frame_result["no_mask_count"],
            "detections": frame_result["detections"],
            "annotated_image": frame_result["annotated_b64"],
            "latency_ms": latency_ms,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("      FACE MASK DETECTION — FLASK WEB SERVER")
    print("=" * 65)
    print("  * Access URL : http://127.0.0.1:5000")
    print("  * Environment: Localhost (CPU Inference)")
    print("=" * 65 + "\n")
    # Pre-warm model
    get_detector()
    app.run(host="127.0.0.1", port=5000, debug=False)
