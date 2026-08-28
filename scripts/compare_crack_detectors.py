"""
compare_crack_detectors.py — render the old and new detector side by side.

WHY: the numbers move on annotated datasets, but the failure that prompted the
rewrite was visible on photographs with no ground truth: confetti on lichen and
grain, and a full-height crack left unmarked. This renders both detectors on
the same image so that specific failure can be checked directly.

The old pipeline is reproduced here rather than imported, because it has been
removed from src/. It is kept verbatim: CLAHE, a 15x15 black-hat, a threshold
that keeps the top 2% of responses, then contour filtering.

Usage:
    python scripts/compare_crack_detectors.py --input-dir data/crack_images/test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.crack_detection.detector import CrackDetector

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def old_mask(image: np.ndarray) -> tuple[np.ndarray, int]:
    """The removed black-hat + fixed-percentile pipeline, for comparison."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
        cv2.GaussianBlur(gray, (5, 5), 0))

    blackhat = cv2.morphologyEx(
        enhanced, cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    cutoff = float(np.percentile(blackhat, 98.0))
    _, mask = cv2.threshold(blackhat, max(cutoff, 1.0), 255, cv2.THRESH_BINARY)

    median = float(np.median(enhanced))
    edges = cv2.dilate(cv2.Canny(enhanced, int(max(0, 0.66 * median)),
                                 int(min(255, 1.33 * median))),
                       cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    mask = cv2.bitwise_or(mask, cv2.bitwise_and(
        edges, cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = np.zeros_like(mask)
    count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        length = cv2.arcLength(contour, closed=False)
        if area < 30 or length < 40:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if (max(w, h) / max(1, min(w, h)) < 3.0
                and (length * length) / max(1.0, area) < 80.0):
            continue
        count += 1
        component = np.zeros_like(mask)
        cv2.drawContours(component, [contour], -1, 255, thickness=cv2.FILLED)
        kept = cv2.bitwise_or(kept, cv2.bitwise_and(component, mask))
    return kept, count


def paint(image: np.ndarray, mask: np.ndarray, caption: str) -> np.ndarray:
    """Red overlay plus a caption bar."""
    painted = image.copy()
    painted[mask > 0] = (0, 0, 255)
    painted = cv2.addWeighted(image, 0.45, painted, 0.55, 0)
    bar = np.zeros((34, painted.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, caption, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([painted, bar])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("data/crack_review/comparison"))
    parser.add_argument("--width", type=int, default=700)
    args = parser.parse_args()

    paths = sorted(p for directory in args.input_dir
                   for p in directory.rglob("*")
                   if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        print(f"No images under {args.input_dir}", file=sys.stderr)
        return 1

    detector = CrackDetector()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'image':44s} {'old count':>10s} {'old mask%':>10s} "
          f"{'new count':>10s} {'new mask%':>10s} {'new sev':>8s}")
    print("-" * 96)
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        scale = args.width / image.shape[1]
        small = cv2.resize(image, (args.width, max(1, int(image.shape[0] * scale))),
                           interpolation=cv2.INTER_AREA)

        old, old_count = old_mask(small)
        new = detector.detect(small)

        old_percent = 100 * float((old > 0).mean())
        new_percent = 100 * float((new.mask > 0).mean())
        print(f"{path.name[:44]:44s} {old_count:>10d} {old_percent:>9.2f}% "
              f"{new.crack_count:>10d} {new_percent:>9.2f}% "
              f"{new.severity_score:>8.3f}")

        panel = np.hstack([
            paint(small, np.zeros(small.shape[:2], np.uint8), f"{path.name[:52]}"),
            paint(small, old, f"OLD  {old_count} regions  {old_percent:.2f}% of pixels"),
            paint(small, (new.mask > 0).astype(np.uint8),
                  f"NEW  {new.crack_count} regions  {new_percent:.2f}%  "
                  f"severity {new.severity_score:.3f}"),
        ])
        cv2.imwrite(str(args.out_dir / f"{path.stem}_compare.jpg"), panel)

    print(f"\nwrote comparison panels to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
