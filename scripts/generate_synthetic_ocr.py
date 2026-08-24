"""Generate licensed-font Ethiopic word images for OCR sequence pretraining."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OCR_IMG_SIZE
from src.ocr.model import GEEZ_CHARSET
from src.ocr.pipeline import resize_pad_image
from src.ocr.training_contract import audit_labels, label_fits_ctc, normalize_ocr_label


def load_corpus(path: Path) -> list[str]:
    labels: list[str] = []
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            labels = [row[1] for row in csv.reader(handle) if len(row) >= 2]
    elif path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            value = payload.get("text") or payload.get("target") or payload.get("original")
            if value:
                labels.append(str(value))
    else:
        labels = path.read_text(encoding="utf-8").splitlines()
    normalized = [normalize_ocr_label(label) for label in labels]
    return [label for label in normalized if label_fits_ctc(label)]


def make_background(height: int, width: int, rng: random.Random) -> np.ndarray:
    base = rng.randint(145, 225)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(4, 18), (height, width))
    background = np.clip(base + noise, 0, 255).astype(np.uint8)
    background = cv2.GaussianBlur(background, (0, 0), rng.uniform(0.4, 1.4))
    return background


def render_label(label: str, font_path: Path, rng: random.Random) -> np.ndarray:
    font_size = rng.randint(24, 42)
    font = ImageFont.truetype(str(font_path), font_size)
    probe = Image.new("L", (1, 1))
    bounds = ImageDraw.Draw(probe).textbbox((0, 0), label, font=font)
    width = max(32, bounds[2] - bounds[0] + rng.randint(18, 40))
    height = max(32, bounds[3] - bounds[1] + rng.randint(12, 24))
    background = make_background(height, width, rng)
    image = Image.fromarray(background)
    draw = ImageDraw.Draw(image)
    x = (width - (bounds[2] - bounds[0])) // 2 - bounds[0]
    y = (height - (bounds[3] - bounds[1])) // 2 - bounds[1]
    ink = rng.randint(15, 75)
    draw.text((x, y), label, font=font, fill=ink)
    array = np.array(image)

    if rng.random() < 0.7:
        kernel = np.ones((rng.choice([1, 2]), rng.choice([1, 2])), np.uint8)
        operation = cv2.erode if rng.random() < 0.5 else cv2.dilate
        array = operation(array, kernel, iterations=1)
    if rng.random() < 0.5:
        array = cv2.GaussianBlur(array, (3, 3), rng.uniform(0.2, 0.9))
    angle = rng.uniform(-7, 7)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, rng.uniform(0.92, 1.08))
    array = cv2.warpAffine(array, matrix, (width, height), borderValue=255)
    return resize_pad_image(array, OCR_IMG_SIZE)


def font_codepoints(path: Path) -> set[int]:
    font = TTFont(path, lazy=True)
    try:
        return {
            codepoint
            for table in font["cmap"].tables
            for codepoint in table.cmap
        }
    finally:
        font.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic Ethiopic OCR sequence data")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--fonts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/geez_synthetic/train_raw"))
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--font-license-manifest", type=Path, required=True)
    args = parser.parse_args()

    fonts = sorted([
        path for path in args.fonts_dir.rglob("*")
        if path.suffix.lower() in {".ttf", ".otf"}
    ])
    if not fonts:
        raise RuntimeError(f"No TTF/OTF fonts found under {args.fonts_dir}")
    license_data = json.loads(args.font_license_manifest.read_text(encoding="utf-8"))
    missing_licenses = [font.name for font in fonts if font.name not in license_data]
    if missing_licenses:
        raise ValueError(f"Fonts missing license metadata: {missing_licenses[:10]}")

    labels = load_corpus(args.corpus)
    audit = audit_labels(labels, GEEZ_CHARSET)
    if not audit.valid:
        raise ValueError(f"Corpus failed OCR label audit: {audit}")
    if not labels:
        raise RuntimeError("Corpus contains no valid CTC labels")

    coverage = {font: font_codepoints(font) for font in fonts}
    fonts_for_label: dict[str, list[Path]] = {}
    for label in set(labels):
        required = {ord(character) for character in label}
        supported = [font for font, codepoints in coverage.items() if required <= codepoints]
        if not supported:
            raise ValueError(f"No licensed font covers every character in label: {label!r}")
        fonts_for_label[label] = supported

    image_dir = args.output_dir / "image_train"
    image_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "image_text_pairs_train.csv"
    rng = random.Random(args.seed)
    rows = []
    font_counts: dict[str, int] = {}
    for index in range(args.samples):
        label = rng.choice(labels)
        font = rng.choice(fonts_for_label[label])
        image = render_label(label, font, rng)
        filename = f"synthetic_{index:07d}.png"
        if not cv2.imwrite(str(image_dir / filename), image):
            raise RuntimeError(f"Failed to write {filename}")
        rows.append((filename, label, "synthetic"))
        font_counts[font.name] = font_counts.get(font.name, 0) + 1

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    provenance = {
        "samples": len(rows),
        "seed": args.seed,
        "corpus": str(args.corpus),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "image_size": list(OCR_IMG_SIZE),
        "fonts": font_counts,
        "font_licenses": license_data,
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Generated {len(rows)} samples at {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
