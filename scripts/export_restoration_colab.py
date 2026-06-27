"""
AXUM — Export Ge'ez restoration JSONL for Google Colab (LLM QLoRA).

Creates a small zip with train/val/eval chat datasets + stats.

Usage (project root, venv active):
    python scripts/generate_dataset.py
    python scripts/export_restoration_colab.py
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from config import (
    GEEZ_RESTORATION_EVAL,
    GEEZ_RESTORATION_TRAIN,
    GEEZ_RESTORATION_VAL,
    ROOT_DIR,
)

EXPORT_DIR = ROOT_DIR / "exports"
ARCHIVE_NAME = "geez_restoration_colab.zip"


def export_restoration_zip(output: Path | None = None) -> Path:
    """
    Pack restoration JSONL files for Colab upload.

    Args:
        output: Zip path (default exports/geez_restoration_colab.zip)

    Returns:
        Path to created archive

    Raises:
        FileNotFoundError: If train JSONL missing (run generate_dataset.py first)
    """
    output = Path(output or EXPORT_DIR / ARCHIVE_NAME)
    files = [
        ("geez_restoration_train.jsonl", GEEZ_RESTORATION_TRAIN),
        ("geez_restoration_val.jsonl", GEEZ_RESTORATION_VAL),
        ("geez_restoration_eval.jsonl", GEEZ_RESTORATION_EVAL),
    ]

    if not GEEZ_RESTORATION_TRAIN.exists():
        raise FileNotFoundError(
            f"Missing {GEEZ_RESTORATION_TRAIN}. Run: python scripts/generate_dataset.py"
        )

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stats_src = GEEZ_RESTORATION_TRAIN.with_name("geez_restoration_dataset.stats.json")

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arc_name, src in files:
            if src.exists():
                zf.write(src, arcname=arc_name)
                logger.info(f"Added {arc_name} ({src.stat().st_size / 1024:.0f} KB)")
        if stats_src.exists():
            zf.write(stats_src, arcname="geez_restoration_dataset.stats.json")

        readme = (
            "AXUM Ge'ez restoration fine-tuning pack\n"
            "Upload to Colab and unzip to /content/\n"
            "Open notebooks/colab_geez_restoration_qlora.ipynb\n"
        )
        zf.writestr("README.txt", readme)

    size_mb = output.stat().st_size / (1024 * 1024)
    logger.info(f"Archive: {output} ({size_mb:.1f} MB)")
    return output


if __name__ == "__main__":
    path = export_restoration_zip()
    print(f"\nUpload to Colab: {path}")
    print("Then open notebooks/colab_geez_restoration_qlora.ipynb")
