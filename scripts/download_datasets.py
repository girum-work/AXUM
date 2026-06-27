"""
AXUM ROVER — Dataset Downloader
=================================
Downloads all training datasets needed for the project.

Datasets:
  1. HHD-Ethiopic  — Ge'ez character images for OCR model training
  2. Artefact images — must be collected manually (script guides you)

Run: python scripts/download_datasets.py

This script downloads ~300-500MB.
Do not interrupt once started.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr.pipeline import download_hhd_ethiopic, verify_dataset
from loguru import logger


def download_geez_ocr_dataset():
    """Download HHD-Ethiopic Ge'ez character dataset."""
    save_dir = "data/geez_characters"

    print("="*60)
    print("DOWNLOADING HHD-ETHIOPIC DATASET")
    print("Source: HuggingFace — OCR-Ethiopic/HHD-Ethiopic")
    print("Size:   ~300-500MB")
    print("="*60)

    try:
        download_hhd_ethiopic(save_dir)
        print("\n✓ Download complete")
    except Exception as e:
        print(f"\n✗ Automatic download failed: {e}")
        print("\nManual fallback:")
        print("1. Go to: https://huggingface.co/datasets/OCR-Ethiopic/HHD-Ethiopic")
        print("2. Click 'Files and versions'")
        print("3. Download parquet files")
        print("4. Extract to: data/geez_characters/")
        print("\nAlternative dataset:")
        print("https://github.com/daniel-yimer/ethiopic-ocr-dataset")
        return False

    print("\nVerifying download...")
    stats = verify_dataset(save_dir)

    if stats.get("total_images", 0) >= 500:
        print("✓ Dataset verified — sufficient for training")
        return True
    else:
        print(" Dataset smaller than expected")
        print("  Consider adding more images manually")
        return False


def check_artefact_dataset():
    """Check artefact classification dataset status."""
    print("\n" + "="*60)
    print("ARTEFACT CLASSIFICATION DATASET")
    print("="*60)
    print("This dataset must be collected manually.")
    print("Target: 50-100 images per category\n")

    classes   = ["pottery", "stone_carving", "coin",
                  "inscription_fragment", "other"]
    base_dir  = Path("data/artefact_classes")
    total     = 0
    ready     = 0

    for cls in classes:
        cls_dir = base_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        images = (list(cls_dir.glob("*.jpg")) +
                  list(cls_dir.glob("*.jpeg")) +
                  list(cls_dir.glob("*.png")))
        count  = len(images)
        total += count
        status = "✓" if count >= 30 else ("⚠" if count >= 10 else "✗")
        if count >= 30:
            ready += 1

        print(f"  {status} {cls:<25} {count:3d} images "
              f"{'(ready)' if count >= 30 else '(need more)' if count >= 10 else '(empty)'}")

    print(f"\nTotal: {total} images | {ready}/{len(classes)} categories ready")

    if total == 0:
        print("\nTo collect artefact images:")
        print("  1. Wikimedia Commons: search 'Aksumite pottery', 'Ethiopian coin'")
        print("  2. British Museum: collections.britishmuseum.org → Search 'Ethiopia'")
        print("  3. Google Images → Tools → Creative Commons")
        print("  Save images directly to data/artefact_classes/<category>/")

    return ready == len(classes)


if __name__ == "__main__":
    print("AXUM ROVER — Dataset Setup\n")

    geez_ok     = download_geez_ocr_dataset()
    artefact_ok = check_artefact_dataset()

    print("\n" + "="*60)
    print("DATASET STATUS SUMMARY")
    print("="*60)
    print(f"  Ge'ez OCR dataset:     {'✓ Ready' if geez_ok else '✗ Needs attention'}")
    print(f"  Artefact dataset:      {'✓ Ready' if artefact_ok else '✗ Collect images'}")
    print("  Restoration dataset:   Run generate_dataset.py to create")

    if geez_ok:
        print("\nNext step: python scripts/train_ocr.py")
    else:
        print("\nNext step: Collect more images, then re-run this script")