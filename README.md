# Face Mask Detection Using Deep Learning

A lightweight, end-to-end Deep Learning system that detects human faces in real time and classifies whether each person is wearing a mask or not (**Mask** vs. **No Mask**).

Designed for college mini-projects to run smoothly on standard laptops with CPU-only training (e.g., Intel Core i5 with 16 GB RAM).

---

## 📌 Project Architecture

```
[ Input Image / Webcam Stream ]
              │
              ▼
   [ OpenCV Face Detection ] ──── (Detects face bounding boxes)
              │
              ▼
    [ Crop Detected Face ]
              │
              ▼
 [ Resize (128x128) & RGB ]
              │
              ▼
   [ Normalization (0 - 1) ]
              │
              ▼
   [ Custom 2D CNN Model ] ────── (Lightweight CNN trained from scratch)
              │
              ▼
   [ Output Classification ] ─── (Mask / No Mask + Confidence Score)
              │
              ▼
    [ Flask Web Interface ] ──── (Interactive UI with live bounding boxes)
```

---

## 📁 Project Directory Structure

```
Face_Mask_Detection/
├── dataset/
│   ├── raw/                 # Raw downloaded dataset (with_mask / without_mask)
│   └── processed/           # Processed & split data (train/val/test) for Stage 2
├── models/                  # Saved CNN weights (.keras) and class label mappings
├── notebooks/               # Exploratory Jupyter Notebooks
├── src/                     # Core Python source package
│   ├── __init__.py
│   ├── config.py            # Centralized project settings & hyperparameters
│   ├── data/                # Data loaders, augmentations, and split helpers
│   ├── training/            # Custom CNN model architecture and train routines
│   └── detection/           # OpenCV face detection and cropping pipeline
├── app/                     # Flask Web Application (Stage 3)
│   ├── templates/           # HTML templates (Jinja2)
│   ├── static/              # CSS, JavaScript, and UI assets
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   └── app.py
├── scripts/
│   └── inspect_dataset.py   # Dataset validation and integrity inspection script
├── tests/                   # Unit tests and pipeline verification tests
├── requirements.txt         # CPU-compatible pinned dependencies
├── .gitignore               # Git ignore rules for virtualenvs, data, and cache
└── README.md                # Project documentation and guide
```

---

## ⚙️ Environment Setup Instructions

Follow these steps to set up your Python environment on Windows:

### 1. Open Terminal and Navigate to Project
Open PowerShell or Command Prompt and enter the project folder:
```powershell
cd Face_Mask_Detection
```

### 2. Create a Virtual Environment
Isolate project packages by creating a virtual environment:
```powershell
python -m venv venv
```

### 3. Activate the Virtual Environment
* **PowerShell**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(Note: If PowerShell shows an Execution Policy error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once).*

* **Command Prompt (CMD)**:
  ```cmd
  venv\Scripts\activate.bat
  ```

Once activated, your terminal prompt will show `(venv)`.

### 4. Install Dependencies
Upgrade pip and install the required CPU-friendly packages:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Dataset Planning & Preparation

### Recommended Dataset: **Face Mask Dataset by Omkar Gurav**
* **Official Kaggle URL**: [https://www.kaggle.com/datasets/omkargurav/face-mask-dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
* **License**: Open Database License (ODbL) / Kaggle Open Community.
* **Download Size**: ~163 MB (compressed ZIP).
* **Classes**: Exactly 2 balanced classes:
  * `with_mask/` (~3,725 cropped face images)
  * `without_mask/` (~3,828 cropped face images)
* **Total Images**: ~7,553 images (RGB).
* **Why this dataset**: Images are already centered on human faces, making it ideal for training a custom CNN without needing complex bounding-box annotations. It fits comfortably in RAM and trains quickly on an Intel Core i5 CPU.

*(Alternative lightweight option: Prajna Bhandary Face Mask Dataset on GitHub / Kaggle — ~30 MB, ~1,376 images).*

---

### Step-by-Step Dataset Placement Instructions

1. Download the archive `face-mask-dataset.zip` from Kaggle.
2. Extract the contents.
3. Place the class folders into `Face_Mask_Detection/dataset/raw/` so that the structure looks like this:

```
Face_Mask_Detection/
└── dataset/
    └── raw/
        ├── with_mask/
        │   ├── 1.jpg
        │   ├── 2.jpg
        │   └── ...
        └── without_mask/
            ├── 1.jpg
            ├── 2.jpg
            └── ...
```

---

## 🔍 Inspecting the Dataset

Once the images are in `dataset/raw/`, run the dataset inspection script:

```powershell
python scripts/inspect_dataset.py
```

The script will automatically:
1. Verify the presence of both classes (`with_mask` and `without_mask`).
2. Count the valid images per class and calculate class balance.
3. Detect any corrupted, zero-byte, or unreadable files using PIL and OpenCV.
4. Compute minimum, maximum, and average image resolutions.
5. Provide recommended sizes for the **70% Training / 15% Validation / 15% Testing** split.

---

---

## 🚀 Stage 2 Execution Guide: Preprocessing, Training & Evaluation

Follow these steps from the `Face_Mask_Detection` directory with your virtual environment activated:

### Step 1: Preprocess and Split the Dataset
Validates images, eliminates duplicates, performs a stratified 70% / 15% / 15% split, and saves manifest files:
```powershell
python src/data/preprocessing.py
```
* **Artifacts Created**:
  * `dataset/processed/train_manifest.csv`
  * `dataset/processed/val_manifest.csv`
  * `dataset/processed/test_manifest.csv`
  * `dataset/processed/split_summary.json`

### Step 2: Run Unit Tests
Verifies label mapping, data splitting logic, CNN architecture output dimensions, and metric computations:
```powershell
python tests/test_stage2.py
```

### Step 3: Train the Custom CNN
Trains the lightweight 4-block CNN on CPU using `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint`:
```powershell
python src/training/train.py --epochs 20 --batch_size 32 --lr 0.001
```
* **Artifacts Created**:
  * `models/best_mask_detector_cnn.keras` (Best weights based on validation loss)
  * `models/class_indices.json` (Numeric label to class mapping)
  * `models/model_metadata.json` (Parameters, input shape, normalization specs)
  * `reports/figures/training_curves.png` (Training & validation loss/accuracy curves)

### Step 4: Evaluate on the Isolated Test Set
Evaluates the saved model strictly on the held-out test set:
```powershell
python src/training/evaluate.py
```
* **Artifacts Created**:
  * `reports/evaluation_metrics.json` (Overall accuracy, macro F1, per-class precision/recall/F1)
  * `reports/figures/confusion_matrix.png` (Test set confusion matrix plot)

---

---

## 🌐 Stage 3: Flask Web Application & Real-Time Detection

The web dashboard integrates OpenCV face detection with the trained CNN model for both image uploads and live webcam video streaming.

### Step 1: Run Stage 3 Tests
Verifies detector initialization, face cropping, zero-face handling, and Flask endpoints:
```powershell
python tests/test_stage3.py
```

### Step 2: Start the Flask Application
Run the local web server:
```powershell
python app/app.py
```

### Step 3: Open in Browser
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

### Features:
1. **Image Upload Analysis**:
   - Drag & drop or select an image file (`JPG`, `PNG`, `WEBP`).
   - Detects all human faces in the image.
   - Draws color-coded bounding boxes:
     - <span style="color:#10B981; font-weight:bold;">Green Box</span>: **MASK** (Wearing Mask)
     - <span style="color:#EF4444; font-weight:bold;">Red Box</span>: **NO MASK** (Not Wearing Mask)
   - Displays prediction confidence percentage for each detected face.
   - Gracefully handles images where no face is detected with an advisory alert.
2. **Live Webcam Detection**:
   - Click **Start Webcam** to start real-time detection through your browser camera.
   - Continuous frame inference at ~10 FPS, optimized for smooth CPU execution.
   - Real-time telemetry displaying inference latency (ms), face count, mask count, and no-mask count.
   - Click **Stop Webcam** to release the camera.

---

## 🗺️ Project Roadmap

- [x] **Stage 1**: Workspace setup, project structure, config, dataset guide, and inspection script.
- [x] **Stage 2**: Data preprocessing & deduplication, stratified 70/15/15 splitting, custom lightweight CNN design, CPU training pipeline with callbacks, test set evaluation metrics, and unit tests.
- [x] **Stage 3**: OpenCV face detection integration, face cropping pipeline, Flask web application backend, responsive dark-mode frontend, and live webcam detection.
