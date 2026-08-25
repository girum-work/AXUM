"""
label_crack_severity.py — rate crack images, then check the detector against you.

WHY: CRACK_SEVERITY_THRESHOLD decides is_treatable, which drives a physical
conservation action, but the number was never checked against human judgement.
The detector's own score is not evidence for itself; something outside the code
has to say what "bad" looks like.

Two modes:

    label      show each image beside the detector's overlay and record a
               1-5 severity rating. Resumable; already-rated images are skipped.

    calibrate  compare stored ratings against detector scores. Reports rank
               correlation, the separation between treatable and not, and the
               threshold that best matches your ratings.

Rating scale, kept coarse on purpose -- finer distinctions are not reliable by
eye and would give false precision:

    1  intact, or hairline marks only
    2  light surface cracking, cosmetic
    3  clear cracking, stable
    4  heavy cracking, structurally concerning
    5  fragmenting or at risk of loss

Usage:
    python scripts/label_crack_severity.py --input-dir data/crack_images
    python scripts/label_crack_severity.py --calibrate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CRACK_SEVERITY_THRESHOLD
from src.crack_detection.detector import CrackDetector

LABEL_PATH = Path("data/crack_labels.json")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# Below this, correlation on so few points is noise rather than signal.
MIN_USEFUL_LABELS = 15

SCALE_HELP = [
    "1  intact / hairline only",
    "2  light surface cracking",
    "3  clear cracking, stable",
    "4  heavy, structurally concerning",
    "5  fragmenting / at risk of loss",
]


def find_images(directories: list[Path]) -> list[Path]:
    """Collect readable images from the given directories."""
    found: list[Path] = []
    for directory in directories:
        if not directory.exists():
            continue
        found.extend(p for p in sorted(directory.rglob("*"))
                     if p.suffix.lower() in IMAGE_SUFFIXES)
    return found


def load_labels() -> dict[str, int]:
    """Read stored ratings, keyed by path relative to the repo root."""
    if not LABEL_PATH.exists():
        return {}
    return json.loads(LABEL_PATH.read_text(encoding="utf-8"))


def save_labels(labels: dict[str, int]) -> None:
    LABEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABEL_PATH.write_text(json.dumps(labels, indent=2, sort_keys=True),
                          encoding="utf-8")


def build_panel(image: np.ndarray, overlay: np.ndarray,
                caption: str, max_width: int = 1500) -> np.ndarray:
    """Put the original beside the detector overlay, with a caption strip."""
    height = min(image.shape[0], 700)
    scale = height / image.shape[0]
    size = (int(image.shape[1] * scale), height)
    panel = np.hstack([cv2.resize(image, size), cv2.resize(overlay, size)])

    if panel.shape[1] > max_width:
        factor = max_width / panel.shape[1]
        panel = cv2.resize(panel, (max_width, int(panel.shape[0] * factor)))

    strip = np.zeros((30 + 22 * len(SCALE_HELP), panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(strip, caption, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    for index, line in enumerate(SCALE_HELP):
        cv2.putText(strip, line, (10, 44 + index * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (170, 170, 170), 1, cv2.LINE_AA)
    return np.vstack([panel, strip])


def run_labelling(directories: list[Path]) -> int:
    """Show each unlabelled image and record a keypress rating."""
    images = find_images(directories)
    if not images:
        print(f"No images under {[str(d) for d in directories]}", file=sys.stderr)
        return 1

    labels = load_labels()
    detector = CrackDetector()
    pending = [p for p in images if str(p.as_posix()) not in labels]

    print(f"{len(images)} images, {len(labels)} already rated, {len(pending)} to go")
    if not pending:
        print("Nothing left to rate. Run with --calibrate.")
        return 0
    print("Keys: 1-5 rate | s skip | u undo last | q quit and save\n")

    order: list[str] = []
    for index, path in enumerate(pending, 1):
        image = cv2.imread(str(path))
        if image is None:
            print(f"  unreadable, skipping: {path}")
            continue

        result = detector.detect(image)
        caption = (f"[{index}/{len(pending)}] {path.name}   "
                   f"detector: {result.severity_score:.3f}  "
                   f"contours: {result.crack_count}")
        cv2.imshow("crack severity", build_panel(image, result.overlay, caption))

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                save_labels(labels)
                print(f"\nSaved {len(labels)} ratings to {LABEL_PATH}")
                return 0
            if key == ord("s"):
                break
            if key == ord("u") and order:
                removed = order.pop()
                labels.pop(removed, None)
                print(f"  undid {removed}")
                break
            if key in (ord(str(n)) for n in range(1, 6)):
                rating = int(chr(key))
                labels[str(path.as_posix())] = rating
                order.append(str(path.as_posix()))
                save_labels(labels)
                print(f"  {path.name}: rated {rating} "
                      f"(detector {result.severity_score:.3f})")
                break

    cv2.destroyAllWindows()
    save_labels(labels)
    print(f"\nSaved {len(labels)} ratings to {LABEL_PATH}")
    return 0


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation, without pulling in scipy."""
    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while (stop + 1 < len(order)
                   and values[order[stop + 1]] == values[order[index]]):
                stop += 1
            shared = (index + stop) / 2 + 1
            for position in range(index, stop + 1):
                ranks[order[position]] = shared
            index = stop + 1
        return ranks

    ra, rb = rank(a), rank(b)
    n = len(a)
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    var_a = sum((x - mean_a) ** 2 for x in ra)
    var_b = sum((y - mean_b) ** 2 for y in rb)
    if var_a == 0 or var_b == 0:
        return 0.0
    return cov / (var_a * var_b) ** 0.5


def run_calibration(treatable_max: int) -> int:
    """Compare stored ratings against detector scores."""
    labels = load_labels()
    if not labels:
        print(f"No ratings yet. Run the labelling mode first.", file=sys.stderr)
        return 1

    detector = CrackDetector()
    rows: list[tuple[str, int, float, int]] = []
    for relative, rating in sorted(labels.items()):
        image = cv2.imread(relative)
        if image is None:
            print(f"  missing, skipping: {relative}")
            continue
        result = detector.detect(image)
        rows.append((Path(relative).name, rating, result.severity_score,
                     result.crack_count))

    if not rows:
        print("No labelled images could be read.", file=sys.stderr)
        return 1

    print(f"\n{'image':38s} {'rated':>6s} {'detector':>9s} {'contours':>9s}")
    print("-" * 66)
    for name, rating, score, count in sorted(rows, key=lambda r: r[1]):
        print(f"{name[:38]:38s} {rating:>6d} {score:>9.3f} {count:>9d}")

    ratings = [float(r[1]) for r in rows]
    scores = [r[2] for r in rows]
    correlation = spearman(ratings, scores)

    print(f"\nlabelled images     : {len(rows)}")
    print(f"rank correlation    : {correlation:+.3f}")

    if len(rows) < MIN_USEFUL_LABELS:
        print(f"\nNOT ENOUGH DATA. {len(rows)} images cannot calibrate a threshold; "
              f"correlation on this few points is mostly noise.")
        print(f"Rate at least {MIN_USEFUL_LABELS}, ideally spread across all five "
              f"levels, then re-run.")
        return 0

    # Sweep candidate thresholds and keep whichever best reproduces the ratings.
    truth = [r[1] <= treatable_max for r in rows]
    best_threshold, best_accuracy = CRACK_SEVERITY_THRESHOLD, -1.0
    for step in range(1, 100):
        candidate = step / 100
        predicted = [s < candidate for s in scores]
        accuracy = sum(p == t for p, t in zip(predicted, truth)) / len(truth)
        if accuracy > best_accuracy:
            best_threshold, best_accuracy = candidate, accuracy

    current = [s < CRACK_SEVERITY_THRESHOLD for s in scores]
    current_accuracy = sum(p == t for p, t in zip(current, truth)) / len(truth)

    print(f"\ntreatable defined as rating <= {treatable_max}")
    print(f"current threshold   : {CRACK_SEVERITY_THRESHOLD:.2f}  "
          f"agrees with you {current_accuracy:.0%} of the time")
    print(f"best threshold      : {best_threshold:.2f}  "
          f"agrees with you {best_accuracy:.0%} of the time")
    if abs(best_threshold - CRACK_SEVERITY_THRESHOLD) > 0.02:
        print(f"\nConsider setting CRACK_SEVERITY_THRESHOLD = {best_threshold:.2f} "
              f"in config.py")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append",
                        help="Repeatable; defaults to the two crack image folders")
    parser.add_argument("--calibrate", action="store_true",
                        help="Compare stored ratings against detector scores")
    parser.add_argument("--treatable-max", type=int, default=2,
                        help="Highest rating still considered treatable")
    args = parser.parse_args()

    if args.calibrate:
        return run_calibration(args.treatable_max)

    directories = args.input_dir or [Path("data/crack_images"),
                                     Path("data/test_crack_detector")]
    return run_labelling(directories)


if __name__ == "__main__":
    raise SystemExit(main())
