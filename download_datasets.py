"""
AXUM ROVER — One-Shot OCR Dataset Download (root shortcut)
==========================================================
WHAT:  Downloads HHD-Ethiopic from HuggingFace into data/geez_characters/.
WHY:   OCR training needs ~80k Ge'ez manuscript images before train_ocr.py runs.
WHEN:  Run ONCE on a fresh clone if data/geez_characters/ is empty or tiny.
       Takes 30–90 minutes — do not close the terminal.

Run:    python download_datasets.py
Prefer: python scripts/download_datasets.py  (full CLI with more options)

After download: python scripts/train_ocr.py
See docs/SETUP.md §9 and docs/CODEBASE_WALKTHROUGH.md §13.
"""

import sys
from pathlib import Path

# Allow `from src.ocr.pipeline import ...` when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.ocr.pipeline import download_hhd_ethiopic, verify_dataset

# Target directory — must match config.py DATA paths and train_ocr.py default
GEEZ_DATA_DIR = "data/geez_characters"


def main() -> None:
    """Download HHD-Ethiopic and verify image count."""
    print("=" * 60)
    print("DOWNLOADING HHD-ETHIOPIC DATASET")
    print("Source: HuggingFace — OCR-Ethiopic/HHD-Ethiopic")
    print("Size: ~300–500MB (full extract is larger on disk)")
    print("=" * 60)

    try:
        download_hhd_ethiopic(GEEZ_DATA_DIR)
        print("\n✓ Download complete")
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nTry: python scripts/download_datasets.py")
        print("Or see docs/SETUP.md §9 for manual fallback.")
        sys.exit(1)

    print("\nVerifying dataset...")
    stats = verify_dataset(GEEZ_DATA_DIR)
    total = stats.get("total_images", 0)

    if total > 500:
        print(f"\n✓ Dataset ready for training ({total} images)")
        print("Next: python scripts/train_ocr.py")
    else:
        print(f"\n✗ Dataset too small ({total} images) — see docs/SETUP.md §9")
        sys.exit(1)


if __name__ == "__main__":
    main()
