"""
Dataset Inspection Script for Face Mask Detection Project.

Inspects the dataset directory to verify:
1. Directory structure and detected class folders.
2. Total image count and per-class balance.
3. Corrupted or unreadable image files (using PIL & OpenCV).
4. Image resolutions (min, max, mean width/height, channel checks).
5. Non-image or zero-byte files.
6. Recommended train/validation/test split sizes.
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Try importing PIL and OpenCV; provide user-friendly message if not yet installed
try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    print("\n[ERROR] Pillow (PIL) is not installed. Please install dependencies:")
    print("        pip install -r requirements.txt\n")
    sys.exit(1)

# Ensure src/config can be imported if script is run directly
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src import config
    DEFAULT_RAW_DIR = config.RAW_DATA_DIR
except Exception:
    DEFAULT_RAW_DIR = PROJECT_ROOT / "dataset" / "raw"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_class_directories(root_dir: Path) -> Dict[str, Path]:
    """
    Locates class subdirectories inside root_dir.
    Handles potential single nested wrapper folders (e.g. raw/Face Mask Dataset/with_mask).
    """
    if not root_dir.exists():
        return {}

    subdirs = [d for d in root_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    # If root_dir contains only one folder and that folder contains subfolders, drill down one level
    if len(subdirs) == 1:
        nested_subdirs = [d for d in subdirs[0].iterdir() if d.is_dir() and not d.name.startswith(".")]
        if len(nested_subdirs) >= 2:
            print(f"[INFO] Detected nested directory wrapper: '{subdirs[0].name}'. Inspecting nested subfolders.")
            return {d.name: d for d in nested_subdirs}

    return {d.name: d for d in subdirs}


def inspect_class_images(class_name: str, class_dir: Path) -> Dict:
    """
    Inspects all images in a class directory.
    Returns counts, dimension statistics, corrupted files, and non-image files.
    """
    total_files = 0
    valid_images = 0
    corrupted_files: List[Tuple[str, str]] = []
    non_image_files: List[str] = []
    widths: List[int] = []
    heights: List[int] = []
    channels_found: set = set()

    all_files = list(class_dir.rglob("*"))

    for file_path in all_files:
        if file_path.is_dir() or file_path.name.startswith("."):
            continue

        total_files += 1
        rel_path = file_path.relative_to(class_dir)

        # Check extension
        if file_path.suffix.lower() not in VALID_EXTENSIONS:
            non_image_files.append(str(rel_path))
            continue

        # Check for zero-byte file
        if file_path.stat().st_size == 0:
            corrupted_files.append((str(rel_path), "Zero-byte file"))
            continue

        # Validate with PIL
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verifies file integrity
            
            # Re-open to read dimensions and mode safely after verify()
            with Image.open(file_path) as img:
                w, h = img.size
                mode = img.mode
                widths.append(w)
                heights.append(h)
                channels_found.add(mode)
                valid_images += 1
        except (UnidentifiedImageError, OSError, Exception) as exc:
            corrupted_files.append((str(rel_path), str(exc)))

    return {
        "class_name": class_name,
        "class_dir": class_dir,
        "total_files": total_files,
        "valid_images": valid_images,
        "corrupted_files": corrupted_files,
        "non_image_files": non_image_files,
        "widths": widths,
        "heights": heights,
        "modes": channels_found,
    }


def format_stats(values: List[int]) -> str:
    """Formats min, max, and average for a list of integers."""
    if not values:
        return "N/A"
    return f"Min: {min(values)}px | Max: {max(values)}px | Mean: {sum(values)/len(values):.1f}px"


def print_inspection_report(raw_dir: Path, results: Dict[str, Dict]):
    """Prints a clear, structured inspection report in the terminal."""
    total_valid = sum(res["valid_images"] for res in results.values())
    total_corrupt = sum(len(res["corrupted_files"]) for res in results.values())
    total_non_image = sum(len(res["non_image_files"]) for res in results.values())
    all_widths = [w for res in results.values() for w in res["widths"]]
    all_heights = [h for res in results.values() for h in res["heights"]]
    all_modes = {m for res in results.values() for m in res["modes"]}

    print("\n" + "=" * 70)
    print("           FACE MASK DETECTION — DATASET INSPECTION REPORT")
    print("=" * 70)
    print(f"Target Directory : {raw_dir}")
    print(f"Detected Classes : {len(results)} ({', '.join(results.keys())})")
    print("-" * 70)

    # Class Breakdown Table
    print(f"{'Class Name':<20} | {'Valid Images':<14} | {'Corrupted':<11} | {'Balance %':<10}")
    print("-" * 70)
    for cls_name, res in results.items():
        count = res["valid_images"]
        pct = (count / total_valid * 100) if total_valid > 0 else 0.0
        corrupt_cnt = len(res["corrupted_files"])
        print(f"{cls_name:<20} | {count:<14} | {corrupt_cnt:<11} | {pct:>6.2f}%")
    print("-" * 70)
    print(f"{'TOTAL':<20} | {total_valid:<14} | {total_corrupt:<11} | 100.00%")
    print("=" * 70)

    # Image Dimension and Color Mode Summary
    print("\n[IMAGE PROPERTIES]")
    print(f"  * Color Modes (PIL)  : {', '.join(sorted(all_modes)) if all_modes else 'N/A'}")
    print(f"  * Image Widths       : {format_stats(all_widths)}")
    print(f"  * Image Heights      : {format_stats(all_heights)}")

    # Corrupted Files Details
    if total_corrupt > 0:
        print("\n[WARNING] Corrupted / Unreadable Files Found:")
        for cls_name, res in results.items():
            for filename, reason in res["corrupted_files"]:
                print(f"  - [{cls_name}] {filename} -> Reason: {reason}")
    else:
        print("\n[INTEGRITY] No corrupted images found! All files readable.")

    # Non-image files
    if total_non_image > 0:
        print("\n[NOTICE] Non-image files found:")
        for cls_name, res in results.items():
            for filename in res["non_image_files"]:
                print(f"  - [{cls_name}] {filename}")

    # Recommended Split Breakdown (70% Train / 15% Val / 15% Test)
    if total_valid > 0:
        train_cnt = int(total_valid * 0.70)
        val_cnt = int(total_valid * 0.15)
        test_cnt = total_valid - train_cnt - val_cnt

        print("\n[RECOMMENDED SPLIT (70% / 15% / 15%)]")
        print(f"  * Training Set   (70%) : ~{train_cnt} images (used for model fitting with augmentation)")
        print(f"  * Validation Set (15%) : ~{val_cnt} images (used for EarlyStopping & checkpointing)")
        print(f"  * Test Set       (15%) : ~{test_cnt} images (held-out for final unbiased evaluation)")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Inspect Face Mask Detection raw dataset.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DEFAULT_RAW_DIR),
        help=f"Path to raw dataset directory (default: {DEFAULT_RAW_DIR})",
    )
    args = parser.parse_args()
    raw_dir = Path(args.data_dir).resolve()

    print(f"\n[SCAN] Searching for dataset in: {raw_dir}")

    if not raw_dir.exists():
        print(f"\n[ERROR] Directory does not exist: {raw_dir}")
        print("        Please ensure you have created the directory or placed the dataset inside it.")
        sys.exit(1)

    class_dirs = find_class_directories(raw_dir)

    if not class_dirs:
        print("\n" + "!" * 70)
        print("[NO DATA FOUND] The raw dataset directory is currently empty!")
        print("!" * 70)
        print("\nTo prepare the dataset:")
        print("1. Download the 'Face Mask Dataset' (e.g. from Kaggle).")
        print("2. Extract the archive into:")
        print(f"   {raw_dir}")
        print("\nExpected directory structure:")
        print(f"   {raw_dir.name}/")
        print("   ├── with_mask/       <- (images of people with masks)")
        print("   └── without_mask/    <- (images of people without masks)")
        print("\n3. Once extracted, re-run this script:")
        print("   python scripts/inspect_dataset.py\n")
        sys.exit(0)

    results = {}
    for cls_name, cls_dir in sorted(class_dirs.items()):
        print(f"[SCAN] Inspecting class folder: '{cls_name}'...")
        results[cls_name] = inspect_class_images(cls_name, cls_dir)

    print_inspection_report(raw_dir, results)


if __name__ == "__main__":
    main()
