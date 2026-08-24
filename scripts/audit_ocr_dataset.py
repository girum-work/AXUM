"""Audit OCR manifests before spending compute on training."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr.model import GEEZ_CHARSET
from src.ocr.training_contract import (
    audit_labels,
    charset_fingerprint,
    min_ctc_timesteps,
    normalize_ocr_label,
)


def audit_manifest(manifest: Path, image_dir: Path, sample_images: int = 1000) -> dict:
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.reader(handle) if len(row) >= 2]
    labels = [normalize_ocr_label(row[1]) for row in rows]
    label_audit = audit_labels(labels, GEEZ_CHARSET)
    sources = Counter(
        row[2]
        if len(row) >= 3 and row[2]
        else "yared" if row[0].startswith("yrd_")
        else "hhd" if row[0].startswith("hhd_") or "geez_characters" in str(manifest)
        else "unknown"
        for row in rows
    )
    lengths = sorted(len(label) for label in labels if label)
    required_steps = sorted(min_ctc_timesteps(label) for label in labels if label)
    missing = []
    dimensions = Counter()
    modes = Counter()
    for row in rows[:sample_images]:
        path = image_dir / row[0].strip()
        if not path.exists():
            missing.append(str(path))
            continue
        with Image.open(path) as image:
            dimensions[f"{image.width}x{image.height}"] += 1
            modes[image.mode] += 1

    def percentile(values: list[int], fraction: float) -> int:
        if not values:
            return 0
        return values[min(len(values) - 1, round((len(values) - 1) * fraction))]

    return {
        "manifest": str(manifest),
        "image_dir": str(image_dir),
        "samples": len(rows),
        "sources": dict(sources),
        "charset_size": len(GEEZ_CHARSET),
        "charset_fingerprint": charset_fingerprint(GEEZ_CHARSET),
        "labels": {
            "valid": label_audit.valid,
            "empty": label_audit.empty,
            "ctc_infeasible": label_audit.ctc_infeasible,
            "unassigned_characters": label_audit.unassigned_characters,
            "unknown_characters": label_audit.unknown_characters,
            "length_p50": percentile(lengths, 0.50),
            "length_p95": percentile(lengths, 0.95),
            "length_max": max(lengths, default=0),
            "ctc_steps_max": max(required_steps, default=0),
        },
        "sampled_images": min(sample_images, len(rows)),
        "missing_images_in_sample": missing,
        "sample_dimensions": dict(dimensions.most_common(20)),
        "sample_modes": dict(modes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit an AXUM OCR CSV manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--sample-images", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    image_dir = args.image_dir or args.manifest.parent / "image_train"
    report = audit_manifest(args.manifest, image_dir, max(0, args.sample_images))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    labels_valid = report["labels"]["valid"]
    images_valid = not report["missing_images_in_sample"]
    return 0 if labels_valid and images_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
