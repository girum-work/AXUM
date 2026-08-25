"""
evaluate_crack_detector.py — score the OpenCV detector against ground truth.

WHY: the detector reports 488-1410 contours on real stone, which looks like
texture rather than damage, but "looks like" is not a measurement. DeepCrack
ships pixel-level masks, so the disagreement can be quantified instead of
argued about.

Reports per-pixel precision, recall, F1 and IoU. Precision is the number that
matters here: low precision with high recall is the signature of over-detection
-- the detector finding every crack and a great deal else besides.

Ground-truth masks are matched to images by stem, so `042.jpg` pairs with
`042.png` regardless of extension. Masks are binarised at >0, and dilated by
`--tolerance` pixels before comparison because hand-traced crack centrelines
are typically a pixel or two off the visible edge; without that slack a correct
detection one pixel wide scores as a miss.

Dataset: Stone331 from https://github.com/qinnzou/DeepCrack (stone surfaces,
331 images). The OneDrive links are folders, so download them by hand:
    Stone331 images : https://1drv.ms/f/s!AittnGm6vRKLtylBkxVXw5arGn6R
    Stone331 GT     : https://1drv.ms/f/s!AittnGm6vRKLwiL55f7f0xdpuD9_
The repository declares no licence; treat as research use and cite the paper.

Usage:
    python scripts/evaluate_crack_detector.py \\
        --images data/deepcrack/stone331/images \\
        --ground-truth data/deepcrack/stone331/gt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.crack_detection.detector import CrackDetector

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def index_by_stem(directory: Path) -> dict[str, Path]:
    """Map filename stem to path, so pairing survives differing extensions."""
    return {p.stem: p for p in sorted(directory.rglob("*"))
            if p.suffix.lower() in IMAGE_SUFFIXES}


def score_pair(predicted: np.ndarray, truth: np.ndarray,
               tolerance: int) -> tuple[float, float, float, float]:
    """
    Compare two binary masks.

    Args:
        predicted: Detector mask, non-zero where crack
        truth: Ground-truth mask, non-zero where crack
        tolerance: Dilation applied to truth before matching

    Returns:
        (precision, recall, f1, iou)
    """
    truth_binary = (truth > 0).astype(np.uint8)
    predicted_binary = (predicted > 0).astype(np.uint8)

    if tolerance > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1))
        truth_matched = cv2.dilate(truth_binary, kernel)
    else:
        truth_matched = truth_binary

    true_positive = float(np.count_nonzero(predicted_binary & truth_matched))
    predicted_total = float(np.count_nonzero(predicted_binary))
    truth_total = float(np.count_nonzero(truth_binary))

    precision = true_positive / predicted_total if predicted_total else 0.0
    # Recall against the undilated truth, so slack cannot inflate it.
    recall_hits = float(np.count_nonzero(
        cv2.dilate(predicted_binary, np.ones((3, 3), np.uint8)) & truth_binary))
    recall = recall_hits / truth_total if truth_total else 0.0

    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    union = float(np.count_nonzero(predicted_binary | truth_binary))
    iou = true_positive / union if union else 0.0
    return precision, recall, f1, iou


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--tolerance", type=int, default=2,
                        help="Pixels of slack allowed around traced centrelines")
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--report", type=Path,
                        default=Path("logs/crack_eval.json"))
    args = parser.parse_args()

    if not args.images.exists() or not args.ground_truth.exists():
        print(f"Missing directory. See this file's docstring for the Stone331 "
              f"download links.", file=sys.stderr)
        return 1

    images = index_by_stem(args.images)
    truths = index_by_stem(args.ground_truth)
    shared = sorted(set(images) & set(truths))
    if not shared:
        print(f"No stems matched between {args.images} and {args.ground_truth}.\n"
              f"  images sample: {list(images)[:4]}\n"
              f"  truth sample : {list(truths)[:4]}", file=sys.stderr)
        return 1
    if args.limit:
        shared = shared[:args.limit]

    print(f"{len(shared)} paired images "
          f"({len(images)} images, {len(truths)} masks)\n")

    detector = CrackDetector()
    rows = []
    for stem in shared:
        image = cv2.imread(str(images[stem]))
        truth = cv2.imread(str(truths[stem]), cv2.IMREAD_GRAYSCALE)
        if image is None or truth is None:
            continue
        if truth.shape[:2] != image.shape[:2]:
            truth = cv2.resize(truth, (image.shape[1], image.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

        result = detector.detect(image)
        precision, recall, f1, iou = score_pair(result.mask, truth, args.tolerance)
        rows.append({
            "stem": stem, "precision": precision, "recall": recall,
            "f1": f1, "iou": iou, "contours": result.crack_count,
            "severity": result.severity_score,
        })

    if not rows:
        print("No readable pairs.", file=sys.stderr)
        return 1

    def mean(key: str) -> float:
        return sum(r[key] for r in rows) / len(rows)

    print(f"{'metric':12s} {'mean':>8s}")
    print("-" * 22)
    for key in ("precision", "recall", "f1", "iou"):
        print(f"{key:12s} {mean(key):>8.3f}")
    print(f"{'contours':12s} {mean('contours'):>8.1f}")
    print(f"{'severity':12s} {mean('severity'):>8.3f}")

    worst = sorted(rows, key=lambda r: r["precision"])[:5]
    print(f"\nlowest precision (most over-detection):")
    for row in worst:
        print(f"  {row['stem'][:28]:28s} P {row['precision']:.3f} "
              f"R {row['recall']:.3f} contours {row['contours']:>5d}")

    if mean("precision") < 0.3 and mean("recall") > 0.5:
        print(f"\nOver-detection confirmed: recall {mean('recall'):.2f} with "
              f"precision {mean('precision'):.2f} means the detector finds the "
              f"cracks and much else besides.")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "images": str(args.images),
        "tolerance": args.tolerance,
        "count": len(rows),
        "mean": {k: mean(k) for k in ("precision", "recall", "f1", "iou")},
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
