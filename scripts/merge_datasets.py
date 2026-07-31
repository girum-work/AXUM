"""
AXUM ROVER — Dataset Merger
============================
Merges two Ge'ez/Ethiopic datasets into a unified training set:

  1. HHD-Ethiopic (word-level, 48k samples, CSV-based)
     Source: data/geez_characters/train_raw/
     Format: image_train/train_XXXXX.png + image_text_pairs_train.csv

  2. Yaredoffice/geez-characters (character-level, 13k samples)
     Source: data/geez_chars_clean/OCR_dataset/train/
     Format: numeric folders 0-286, each containing PNG images
     Mapping: folder N -> chr(0x1200 + N)

Output: data/geez_merged/
  train_raw/
    image_train/     <- all images from both datasets
    image_text_pairs_train.csv  <- unified CSV

The merged dataset is drop-in compatible with HHDEthiopicDataset
in pipeline.py — no code changes needed.

Run: python scripts/merge_datasets.py
"""

import csv
import shutil
import sys
from pathlib import Path

from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Paths ──────────────────────────────────────────────────────
HHD_DIR     = Path("data/geez_characters/train_raw")
YARED_DIR   = Path("data/geez_chars_clean/OCR_dataset/train")
OUTPUT_DIR  = Path("data/geez_merged/train_raw")
IMG_OUT_DIR = OUTPUT_DIR / "image_train"
CSV_OUT     = OUTPUT_DIR / "image_text_pairs_train.csv"

# Yaredoffice folder index -> Ge'ez character
# Folder 0 = U+1200 (ሀ), folder 1 = U+1201 (ሁ), etc.
YARED_INDEX_TO_CHAR = {i: chr(0x1200 + i) for i in range(287)}


def merge_datasets():
    """
    Main merge function. Copies all images into a single folder
    and writes a unified CSV mapping filename -> Ge'ez label.
    """
    logger.info("="*60)
    logger.info("AXUM — Dataset Merger")
    logger.info("="*60)

    from src.ocr.pipeline import label_fits_ctc

    # ── Setup output dirs ──────────────────────────────────────
    IMG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {OUTPUT_DIR}")

    rows = []  # (filename, label) pairs for the CSV

    # ══════════════════════════════════════════════════════════
    # SOURCE 1: HHD-Ethiopic
    # ══════════════════════════════════════════════════════════
    logger.info("\nSource 1: HHD-Ethiopic (word-level)")

    hhd_csv  = HHD_DIR / "image_text_pairs_train.csv"
    hhd_imgs = HHD_DIR / "image_train"

    if not hhd_csv.exists():
        logger.error(f"HHD CSV not found: {hhd_csv}")
        logger.error("Run download_datasets.py first")
        return False

    hhd_count   = 0
    hhd_skipped = 0

    with open(hhd_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        hhd_rows = list(reader)

    logger.info(f"  CSV rows: {len(hhd_rows)}")

    for row in tqdm(hhd_rows, desc="  Copying HHD"):
        if len(row) < 2:
            hhd_skipped += 1
            continue

        fname = row[0].strip()
        label = row[1].strip()

        if not label or not fname:
            hhd_skipped += 1
            continue

        # Skip labels that cannot fit in the model's 26-timestep CTC budget
        if not label_fits_ctc(label):
            hhd_skipped += 1
            continue

        src = hhd_imgs / fname
        if not src.exists():
            hhd_skipped += 1
            continue

        # Prefix to avoid filename collisions with Yaredoffice
        new_fname = "hhd_" + fname
        dst = IMG_OUT_DIR / new_fname

        shutil.copy2(src, dst)
        rows.append((new_fname, label))
        hhd_count += 1

    logger.info(f"  Copied: {hhd_count} | Skipped: {hhd_skipped}")

    # ══════════════════════════════════════════════════════════
    # SOURCE 2: Yaredoffice (character-level)
    # ══════════════════════════════════════════════════════════
    logger.info("\nSource 2: Yaredoffice (character-level)")

    if not YARED_DIR.exists():
        logger.warning(f"Yaredoffice dir not found: {YARED_DIR}")
        logger.warning("Skipping — merge will use HHD only")
    else:
        yared_count   = 0
        yared_skipped = 0
        yared_unknown = 0

        char_dirs = sorted(
            YARED_DIR.iterdir(),
            key=lambda p: int(p.name) if p.name.isdigit() else 9999
        )

        for char_dir in tqdm(char_dirs, desc="  Copying Yaredoffice"):
            if not char_dir.is_dir():
                continue

            # Map numeric folder name to Ge'ez character
            if not char_dir.name.isdigit():
                yared_unknown += 1
                continue

            folder_idx = int(char_dir.name)
            if folder_idx not in YARED_INDEX_TO_CHAR:
                yared_unknown += 1
                continue

            char = YARED_INDEX_TO_CHAR[folder_idx]

            # Copy all images in this character folder
            images = (list(char_dir.glob("*.png")) +
                      list(char_dir.glob("*.jpg")) +
                      list(char_dir.glob("*.jpeg")))

            for img_path in images:
                new_fname = f"yrd_{folder_idx:03d}_{img_path.name}"
                dst = IMG_OUT_DIR / new_fname

                try:
                    shutil.copy2(img_path, dst)
                    rows.append((new_fname, char))
                    yared_count += 1
                except Exception as e:
                    logger.debug(f"Copy failed {img_path}: {e}")
                    yared_skipped += 1

        logger.info(f"  Copied: {yared_count} | "
                   f"Skipped: {yared_skipped} | "
                   f"Unknown folders: {yared_unknown}")

    # ══════════════════════════════════════════════════════════
    # Write unified CSV
    # ══════════════════════════════════════════════════════════
    logger.info(f"\nWriting CSV: {CSV_OUT}")
    logger.info(f"Total samples: {len(rows)}")

    with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    logger.info("CSV written")

    # ── Verification ───────────────────────────────────────────
    logger.info("\nVerifying merged dataset...")

    sample_count = sum(
        1 for p in IMG_OUT_DIR.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )

    hhd_samples   = sum(1 for r in rows if r[0].startswith("hhd_"))
    yared_samples = sum(1 for r in rows if r[0].startswith("yrd_"))

    print("\n" + "="*60)
    print("MERGE COMPLETE")
    print("="*60)
    print(f"  HHD-Ethiopic samples:    {hhd_samples:,}")
    print(f"  Yaredoffice samples:     {yared_samples:,}")
    print(f"  Total CSV rows:          {len(rows):,}")
    print(f"  Image files on disk:     {sample_count:,}")
    print(f"  Output directory:        {OUTPUT_DIR.resolve()}")
    print("="*60)
    print(f"\nNext step: update train_ocr.py data_dir to:")
    print(f"  data/geez_merged")
    print(f"Or retrain on Colab using the merged dataset.")

    return True


def verify_merged_dataset():
    """
    Quick verification that the merged dataset loads correctly
    with the existing HHDEthiopicDataset class.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        from src.ocr.pipeline import HHDEthiopicDataset
        ds = HHDEthiopicDataset("data/geez_merged", split="train")
        print(f"\nDataset loads correctly: {len(ds)} training samples")

        if len(ds) > 0:
            img, lbl, llen = ds[0]
            print(f"Image shape:  {img.shape}")
            print(f"Label length: {llen.item()}")
            print("\nReady for Colab training run.")
        else:
            print("WARNING: 0 samples loaded — check merger output")

    except Exception as e:
        print(f"Verification failed: {e}")


if __name__ == "__main__":
    import sys

    success = merge_datasets()

    if success:
        print("\nRunning verification...")
        verify_merged_dataset()