"""
AXUM — Export Ge'ez OCR training data for Google Colab.

Creates a zip archive with the folder layout expected by HHDEthiopicDataset:
    <data_name>/train_raw/image_text_pairs_train.csv
    <data_name>/train_raw/image_train/*.png

Usage (from project root, venv active):
    python scripts/export_colab_training_data.py
    python scripts/export_colab_training_data.py --data-root data/geez_merged -o exports/geez_merged_colab.zip
    python scripts/export_colab_training_data.py --include-checkpoint
    python scripts/export_colab_training_data.py --include-code
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

ROOT = Path(__file__).parent.parent
CSV_NAME = "image_text_pairs_train.csv"
IMG_DIR = "image_train"
EXPORT_DIR = ROOT / "exports"

CODE_FILES = [
    "config.py",
    "scripts/train_ocr.py",
    "src/ocr/model.py",
    "src/ocr/pipeline.py",
    "src/ocr/__init__.py",
    "src/__init__.py",
]


def _add_tree(
    zf: zipfile.ZipFile,
    source_dir: Path,
    arc_prefix: str,
    extensions: set[str] | None = None,
) -> int:
    """
    Add all files under source_dir to the zip with a given archive prefix.

    Returns:
        Number of files added.
    """
    count = 0
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if extensions and path.suffix.lower() not in extensions:
            continue
        arcname = f"{arc_prefix}/{path.relative_to(source_dir).as_posix()}"
        zf.write(path, arcname=arcname)
        count += 1
    return count


def verify_zip_archive(zip_path: Path, data_name: str) -> dict:
    """
    Validate a Colab data zip: readable archive, CSV rows, images on disk in zip.

    Returns:
        Stats dict with counts and any error messages.
    """
    import csv
    import zipfile

    stats: dict = {
        "ok": False,
        "errors": [],
        "csv_rows": 0,
        "zip_images": 0,
        "csv_images_referenced": 0,
        "missing_in_zip": 0,
    }
    prefix = f"{data_name}/train_raw"
    csv_arc = f"{prefix}/{CSV_NAME}"
    img_prefix = f"{prefix}/{IMG_DIR}/"

    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            if csv_arc not in names:
                stats["errors"].append(f"Missing {csv_arc}")
                return stats

            with zf.open(csv_arc) as f:
                rows = list(csv.reader(line.decode("utf-8") for line in f))
            stats["csv_rows"] = len(rows)
            stats["csv_images_referenced"] = sum(1 for r in rows if len(r) >= 2 and r[0].strip())

            zip_images = {n for n in names if n.startswith(img_prefix) and n.lower().endswith((".png", ".jpg", ".jpeg"))}
            stats["zip_images"] = len(zip_images)

            missing = 0
            for row in rows:
                if len(row) < 2 or not row[0].strip():
                    continue
                arc = img_prefix + row[0].strip().replace("\\", "/")
                if arc not in names:
                    missing += 1
            stats["missing_in_zip"] = missing
            stats["ok"] = (
                stats["csv_rows"] > 0
                and stats["zip_images"] > 0
                and missing == 0
                and abs(stats["zip_images"] - stats["csv_images_referenced"]) <= stats["csv_rows"] * 0.01
            )
    except zipfile.BadZipFile:
        stats["errors"].append("Not a valid zip file")
    except Exception as e:
        stats["errors"].append(str(e))

    return stats


def export_data_zip(
    output_path: Path,
    data_root: Path,
    include_checkpoint: bool,
    include_code: bool,
) -> None:
    """
    Build the Colab upload archive(s).

    Args:
        output_path: Path for the main data zip file.
        data_root: e.g. data/geez_merged or data/geez_characters
        include_checkpoint: Also pack models/geez_ocr.pth if it exists.
        include_code: Also pack a separate code zip for Colab.
    """
    train_raw = data_root / "train_raw"
    csv_path = train_raw / CSV_NAME
    img_dir = train_raw / IMG_DIR
    data_name = data_root.name

    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing image folder: {img_dir}")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{data_name}/train_raw"

    logger.info(f"Packing {csv_path.name} and {IMG_DIR}/ ...")
    with zipfile.ZipFile(
        output_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as zf:
        zf.write(csv_path, arcname=f"{prefix}/{CSV_NAME}")
        n_images = _add_tree(zf, img_dir, f"{prefix}/{IMG_DIR}", {".png", ".jpg", ".jpeg"})

        if include_checkpoint:
            ckpt = ROOT / "models" / "geez_ocr.pth"
            if ckpt.is_file():
                zf.write(ckpt, arcname="models/geez_ocr.pth")
                logger.info(f"Included checkpoint: {ckpt}")
            else:
                logger.warning("No checkpoint at models/geez_ocr.pth — skipped")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nData archive: {output_path}")
    print(f"  Images:     {n_images}")
    print(f"  Size:       {size_mb:.1f} MB")
    print(f"\nIn Colab, after upload:")
    print(f"  !unzip -q {output_path.name} -d /content/data")
    print(f"  # -> /content/data/{data_name}/train_raw/ ...")
    print(f"  # train with: data_dir='data/{data_name}'")

    if include_code:
        code_zip = EXPORT_DIR / "geez_ocr_colab_code.zip"
        with zipfile.ZipFile(code_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for rel in CODE_FILES:
                src = ROOT / rel
                if src.is_file():
                    zf.write(src, arcname=rel.replace("\\", "/"))
        code_mb = code_zip.stat().st_size / (1024 * 1024)
        print(f"\nCode archive: {code_zip} ({code_mb:.1f} MB)")
        print("  !unzip -q geez_ocr_colab_code.zip -d /content/axum")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Ge'ez OCR training data for Colab")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "geez_characters",
        help="Dataset root containing train_raw/ (default: data/geez_characters)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output zip path (default: exports/<data_name>_colab.zip)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify an existing zip (use with -o)",
    )
    parser.add_argument(
        "--include-checkpoint",
        action="store_true",
        help="Include models/geez_ocr.pth from local epoch-1 run",
    )
    parser.add_argument(
        "--include-code",
        action="store_true",
        help="Also create exports/geez_ocr_colab_code.zip",
    )
    args = parser.parse_args()

    output = args.output or (EXPORT_DIR / f"{args.data_root.name}_colab.zip")

    if args.verify_only:
        stats = verify_zip_archive(output, args.data_root.name)
        print(stats)
        sys.exit(0 if stats.get("ok") else 1)

    export_data_zip(output, args.data_root, args.include_checkpoint, args.include_code)

    stats = verify_zip_archive(output, args.data_root.name)
    print("\nVerification:")
    print(f"  OK:              {stats['ok']}")
    print(f"  CSV rows:        {stats['csv_rows']:,}")
    print(f"  Images in zip:   {stats['zip_images']:,}")
    if stats["errors"]:
        print(f"  Errors:          {stats['errors']}")
    if not stats["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
