"""
score_crack_grids.py — score cell-level crack readings against ground truth.

WHY: before distilling a reader's judgement into a model, measure whether that
judgement is actually better than the filters it would replace. Both are scored
on the same task -- which of 64 cells does a crack pass through -- so the
comparison is like for like. Pixel F1 would not be, because a reader cannot
produce a pixel mask.

Usage:
    python scripts/score_crack_grids.py --answers data/crack_grids/reading.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

logger.remove()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.crack_detection.detector import CrackDetector
from scripts.render_crack_grids import DATASETS, index_by_stem, truth_cells


def score(predicted: set[str], truth: set[str]) -> tuple[float, float, float]:
    hits = len(predicted & truth)
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(truth) if truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return precision, recall, f1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path,
                        default=Path("data/crack_grids/reading.json"))
    parser.add_argument("--divisions", type=int, default=8)
    args = parser.parse_args()

    readings = json.loads(args.answers.read_text(encoding="utf-8"))
    detector = CrackDetector()

    print(f"{'image':32s} {'truth':>6s} | {'reader P':>8s} {'R':>6s} {'F1':>6s} "
          f"| {'detect P':>8s} {'R':>6s} {'F1':>6s}")
    print("-" * 88)

    reader_scores, detector_scores = [], []
    for key, cells in readings.items():
        dataset, stem = key.split("/", 1)
        image_dir, truth_dir = DATASETS[dataset]
        image = cv2.imread(str(index_by_stem(image_dir)[stem]))
        mask = cv2.imread(str(index_by_stem(truth_dir)[stem]), cv2.IMREAD_GRAYSCALE)

        truth = set(truth_cells(mask, args.divisions))
        reader = set(cells)
        detected = set(truth_cells(detector.detect(image).mask, args.divisions))

        reader_scores.append(score(reader, truth))
        detector_scores.append(score(detected, truth))
        print(f"{stem[:32]:32s} {len(truth):>6d} | "
              f"{reader_scores[-1][0]:>8.2f} {reader_scores[-1][1]:>6.2f} "
              f"{reader_scores[-1][2]:>6.2f} | "
              f"{detector_scores[-1][0]:>8.2f} {detector_scores[-1][1]:>6.2f} "
              f"{detector_scores[-1][2]:>6.2f}")

    def mean(scores, index):
        return sum(s[index] for s in scores) / len(scores)

    print("-" * 88)
    print(f"{'MEAN':32s} {'':>6s} | {mean(reader_scores, 0):>8.2f} "
          f"{mean(reader_scores, 1):>6.2f} {mean(reader_scores, 2):>6.2f} | "
          f"{mean(detector_scores, 0):>8.2f} {mean(detector_scores, 1):>6.2f} "
          f"{mean(detector_scores, 2):>6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
