"""
Test Ge'ez OCR on HHD-Ethiopic training crops (direct predict, no region detection).

HHD images are already cropped text strips — process_image() region detection
often finds zero boxes on them. This script calls model.predict() directly.

Run from project root:
    python scripts/test_ocr_samples.py
    python scripts/test_ocr_samples.py --limit 10
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OCR_MODEL_PATH
from src.ocr.model import load_ocr_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Test OCR on HHD sample images")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("data/geez_characters/train_raw/image_train"),
        help="Folder of PNG/JPG crops",
    )
    parser.add_argument("--limit", type=int, default=3, help="Max images to test")
    parser.add_argument(
        "--model",
        type=Path,
        default=OCR_MODEL_PATH,
        help="Path to geez_ocr.pth",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}")
        print("Download geez_ocr.pth from Colab into models/")
        sys.exit(1)

    if not args.dir.exists():
        print(f"Image dir not found: {args.dir}")
        sys.exit(1)

    images = sorted(args.dir.glob("*.png")) + sorted(args.dir.glob("*.jpg"))
    if not images:
        print(f"No images in {args.dir}")
        sys.exit(1)

    print(f"Loading {args.model} ...")
    model = load_ocr_model(args.model)

    # UTF-8 stdout on Windows so Ge'ez prints correctly
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"Testing {min(args.limit, len(images))} of {len(images)} images\n")

    for img_path in images[: args.limit]:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"{img_path.name}: failed to read")
            continue

        result = model.predict(img)
        text = result["text"] or "(empty)"
        conf = result["confidence"]
        print(f"{img_path.name}: [{text}] conf={conf:.2f}")


if __name__ == "__main__":
    main()
