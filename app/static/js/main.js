/**
 * FaceShield CNN — Client-side Application Controller
 * Handles image upload drag-and-drop, API requests, and live webcam canvas streaming.
 */

// Global State
let currentSelectedFile = null;
let webcamStream = null;
let isWebcamRunning = false;
let webcamIntervalId = null;
let isFrameProcessing = false;

// DOM Elements
const tabUploadBtn = document.getElementById("tabUploadBtn");
const tabWebcamBtn = document.getElementById("tabWebcamBtn");
const panelUpload = document.getElementById("panelUpload");
const panelWebcam = document.getElementById("panelWebcam");

// Upload Elements
const dropzone = document.getElementById("dropzone");
const dropzonePrompt = document.getElementById("dropzonePrompt");
const imageInput = document.getElementById("imageInput");
const uploadPreview = document.getElementById("uploadPreview");
const analyzeBtn = document.getElementById("analyzeBtn");
const resetBtn = document.getElementById("resetBtn");
const resultImage = document.getElementById("resultImage");
const emptyState = document.getElementById("emptyState");
const uploadSpinner = document.getElementById("uploadSpinner");
const statsRow = document.getElementById("statsRow");
const countTotal = document.getElementById("countTotal");
const countMask = document.getElementById("countMask");
const countNoMask = document.getElementById("countNoMask");
const noFaceAlert = document.getElementById("noFaceAlert");
const uploadLatency = document.getElementById("uploadLatency");
const detectionsList = document.getElementById("detectionsList");
const chipsContainer = document.getElementById("chipsContainer");

// Webcam Elements
const webcamVideo = document.getElementById("webcamVideo");
const captureCanvas = document.getElementById("captureCanvas");
const webcamStreamOutput = document.getElementById("webcamStreamOutput");
const webcamPlaceholder = document.getElementById("webcamPlaceholder");
const webcamSpinner = document.getElementById("webcamSpinner");
const startCamBtn = document.getElementById("startCamBtn");
const stopCamBtn = document.getElementById("stopCamBtn");
const webcamStatusPill = document.getElementById("webcamStatusPill");
const webcamStatusText = document.getElementById("webcamStatusText");
const liveFps = document.getElementById("liveFps");
const liveFaceCount = document.getElementById("liveFaceCount");
const liveMaskCount = document.getElementById("liveMaskCount");
const liveNoMaskCount = document.getElementById("liveNoMaskCount");

// ==========================================================================
// Mode Switcher (Upload vs. Webcam)
// ==========================================================================
function switchMode(mode) {
    if (mode === "upload") {
        tabUploadBtn.classList.add("active");
        tabWebcamBtn.classList.remove("active");
        panelUpload.classList.add("active");
        panelWebcam.classList.remove("active");
        // Ensure camera is stopped if switching to upload mode
        if (isWebcamRunning) {
            stopWebcam();
        }
    } else {
        tabWebcamBtn.classList.add("active");
        tabUploadBtn.classList.remove("active");
        panelWebcam.classList.add("active");
        panelUpload.classList.remove("active");
    }
}

// ==========================================================================
// Image Upload & Drag-and-Drop
// ==========================================================================
dropzone.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
        handleFileSelect(e.target.files[0]);
    }
});

["dragenter", "dragover"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });
});

["dragleave", "drop"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
    });
});

dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

function handleFileSelect(file) {
    if (!file.type.match("image.*")) {
        alert("Please select a valid image file (JPEG, PNG, WEBP).");
        return;
    }

    currentSelectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        uploadPreview.src = e.target.result;
        uploadPreview.style.display = "block";
        dropzonePrompt.style.display = "none";
        analyzeBtn.disabled = false;
        resetBtn.style.display = "inline-flex";
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    currentSelectedFile = null;
    imageInput.value = "";
    uploadPreview.style.display = "none";
    dropzonePrompt.style.display = "block";
    analyzeBtn.disabled = true;
    resetBtn.style.display = "none";

    resultImage.style.display = "none";
    emptyState.style.display = "block";
    statsRow.style.display = "none";
    noFaceAlert.style.display = "none";
    uploadLatency.style.display = "none";
    detectionsList.style.display = "none";
    chipsContainer.innerHTML = "";
}

async function analyzeUploadedImage() {
    if (!currentSelectedFile) return;

    analyzeBtn.disabled = true;
    uploadSpinner.style.display = "block";
    emptyState.style.display = "none";
    resultImage.style.display = "none";
    noFaceAlert.style.display = "none";
    statsRow.style.display = "none";
    detectionsList.style.display = "none";

    const formData = new FormData();
    formData.append("image", currentSelectedFile);

    try {
        const response = await fetch("/api/predict-image", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        uploadSpinner.style.display = "none";
        analyzeBtn.disabled = false;

        if (!data.success) {
            alert(`Error: ${data.error || "Failed to process image."}`);
            emptyState.style.display = "block";
            return;
        }

        // Render result image
        resultImage.src = data.annotated_image;
        resultImage.style.display = "block";

        // Latency
        uploadLatency.textContent = `Latency: ${data.latency_ms}ms`;
        uploadLatency.style.display = "inline-block";

        // Counts
        countTotal.textContent = data.faces_detected;
        countMask.textContent = data.mask_count;
        countNoMask.textContent = data.no_mask_count;
        statsRow.style.display = "grid";

        // Handle Zero Faces Detected
        if (data.faces_detected === 0) {
            noFaceAlert.style.display = "flex";
        } else {
            noFaceAlert.style.display = "none";
            // Render Detection Chips
            chipsContainer.innerHTML = "";
            data.detections.forEach((det, idx) => {
                const chip = document.createElement("div");
                const isMask = det.label === "MASK";
                chip.className = `detection-chip ${isMask ? "mask" : "nomask"}`;
                chip.innerHTML = `
                    <span>Face ${idx + 1}:</span>
                    <strong>${det.label}</strong>
                    <span>(${det.confidence_percent})</span>
                `;
                chipsContainer.appendChild(chip);
            });
            detectionsList.style.display = "block";
        }

    } catch (err) {
        uploadSpinner.style.display = "none";
        analyzeBtn.disabled = false;
        emptyState.style.display = "block";
        alert(`Request failed: ${err.message}`);
    }
}

// ==========================================================================
// Live Webcam Streaming
// ==========================================================================
async function startWebcam() {
    try {
        startCamBtn.disabled = true;
        webcamSpinner.style.display = "block";
        webcamPlaceholder.style.display = "none";

        // Request camera with 640x480 resolution (optimal CPU throughput)
        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            },
            audio: false,
        };

        webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
        webcamVideo.srcObject = webcamStream;
        await webcamVideo.play();

        // Canvas setup for frame capture
        captureCanvas.width = 640;
        captureCanvas.height = 480;

        webcamSpinner.style.display = "none";
        webcamStreamOutput.style.display = "block";
        stopCamBtn.disabled = false;
        webcamStatusPill.classList.add("live");
        webcamStatusText.textContent = "Camera Live";
        isWebcamRunning = true;

        // Process frames at ~10 FPS (100ms interval) to maintain silky CPU performance
        webcamIntervalId = setInterval(captureAndSendFrame, 100);

    } catch (err) {
        webcamSpinner.style.display = "none";
        webcamPlaceholder.style.display = "block";
        startCamBtn.disabled = false;
        alert(`Camera access denied or unavailable: ${err.message}`);
    }
}

function stopWebcam() {
    isWebcamRunning = false;
    if (webcamIntervalId) {
        clearInterval(webcamIntervalId);
        webcamIntervalId = null;
    }

    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }

    webcamVideo.pause();
    webcamVideo.srcObject = null;

    webcamStreamOutput.style.display = "none";
    webcamPlaceholder.style.display = "block";
    startCamBtn.disabled = false;
    stopCamBtn.disabled = true;
    webcamStatusPill.classList.remove("live");
    webcamStatusText.textContent = "Camera Offline";

    liveFps.textContent = "-- ms";
    liveFaceCount.textContent = "0";
    liveMaskCount.textContent = "0";
    liveNoMaskCount.textContent = "0";
}

async function captureAndSendFrame() {
    if (!isWebcamRunning || isFrameProcessing) return;

    const ctx = captureCanvas.getContext("2d");
    if (!ctx || webcamVideo.videoWidth === 0) return;

    isFrameProcessing = true;
    ctx.drawImage(webcamVideo, 0, 0, captureCanvas.width, captureCanvas.height);
    const frameBase64 = captureCanvas.toDataURL("image/jpeg", 0.75);

    try {
        const response = await fetch("/api/predict-frame", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ frame: frameBase64 }),
        });

        if (response.ok) {
            const data = await response.json();
            if (data.success && isWebcamRunning) {
                webcamStreamOutput.src = data.annotated_image;
                liveFps.textContent = `${data.latency_ms} ms`;
                liveFaceCount.textContent = data.faces_detected;
                liveMaskCount.textContent = data.mask_count;
                liveNoMaskCount.textContent = data.no_mask_count;
            }
        }
    } catch (e) {
        // Drop transient frame errors silently during stream
    } finally {
        isFrameProcessing = false;
    }
}
