"""
label_crack_severity.py — rate crack images, then check the detector against you.

WHY: CRACK_SEVERITY_THRESHOLD decides is_treatable, which drives a physical
conservation action, but the number was never checked against human judgement.
The detector's own score is not evidence for itself; something outside the code
has to say what "bad" looks like.

Two modes:

    label      write a review panel per image (original beside the detector
               overlay) and take a 1-5 rating from the terminal. Resumable;
               already-rated images are skipped. No GUI window is used, because
               requirements.txt pins opencv-python-headless.

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
    python scripts/label_crack_severity.py --panels-only --stratified 30
    python scripts/label_crack_severity.py --calibrate
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CRACK_SEVERITY_THRESHOLD
from src.crack_detection.detector import CrackDetector

LABEL_PATH = Path("data/crack_labels.json")
# Records which images the stratified sampler chose, so rating can be stopped
# and resumed without re-sampling a different set and rewriting the panels.
SELECTION_PATH = Path("data/crack_review/selection.json")
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


def stratified_sample(images: list[Path], detector: CrackDetector, count: int,
                      bin_count: int = 6, scan_limit: int = 400) -> list[Path]:
    """
    Pick images spread across the detector's severity range.

    Random sampling wastes ratings: artefact photos cluster heavily in one band,
    so 30 random images can only locate the threshold inside that band. Spanning
    the range puts each rating where it carries information about the boundary.

    Args:
        images: Candidate paths
        detector: Used only to spread the sample, never as ground truth
        count: How many images to return
        bin_count: Severity bands to spread across
        scan_limit: Cap on images scored, to keep selection quick

    Returns:
        Selected paths, ordered from least to most severe
    """
    candidates = images if len(images) <= scan_limit else random.sample(images, scan_limit)

    scored: list[tuple[float, Path]] = []
    for path in candidates:
        image = cv2.imread(str(path))
        if image is None:
            continue
        # Large museum photographs are slow and the score is scale-sensitive.
        if max(image.shape[:2]) > 1200:
            scale = 1200 / max(image.shape[:2])
            image = cv2.resize(image, (int(image.shape[1] * scale),
                                       int(image.shape[0] * scale)))
        scored.append((detector.detect(image).severity_score, path))

    if not scored:
        return []

    buckets: dict[int, list[tuple[float, Path]]] = {}
    for score, path in scored:
        index = min(bin_count - 1, int(score * bin_count))
        buckets.setdefault(index, []).append((score, path))

    selected: list[tuple[float, Path]] = []
    per_bin = max(1, count // max(len(buckets), 1))
    for index in sorted(buckets):
        pool = buckets[index]
        random.shuffle(pool)
        selected.extend(pool[:per_bin])

    # Top up from whatever is left if some bands were sparse.
    if len(selected) < count:
        chosen = {p for _, p in selected}
        remaining = [(s, p) for s, p in scored if p not in chosen]
        random.shuffle(remaining)
        selected.extend(remaining[:count - len(selected)])

    selected.sort(key=lambda item: item[0])
    return [path for _score, path in selected[:count]]


def run_labelling(directories: list[Path], panel_dir: Path,
                  stratified: int = 0, panels_only: bool = False) -> int:
    """
    Write review panels to disk, then take ratings from the terminal.

    No cv2.imshow: requirements.txt pins opencv-python-headless, which has no
    GUI at all. Writing files also means this works over SSH and on the Pi.
    """
    labels = load_labels()

    # Panels already prepared: rate those rather than scoring the corpus again.
    if not stratified and not panels_only and SELECTION_PATH.exists():
        selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
        written = [(Path(entry["image"]), Path(entry["panel"]),
                    entry["score"], entry["contours"])
                   for entry in selection
                   if entry["image"] not in labels
                   and Path(entry["panel"]).exists()]
        if written:
            print(f"Resuming {SELECTION_PATH}: {len(selection)} panels, "
                  f"{len(selection) - len(written)} already rated, "
                  f"{len(written)} to go")
            return prompt_for_ratings(written, labels, panel_dir)
        print(f"Every panel in {SELECTION_PATH} is rated. Run with --calibrate.")
        return 0

    images = find_images(directories)
    if not images:
        print(f"No images under {[str(d) for d in directories]}", file=sys.stderr)
        return 1

    detector = CrackDetector()
    pending = [p for p in images if str(p.as_posix()) not in labels]

    print(f"{len(images)} images, {len(labels)} already rated, {len(pending)} to go")
    if not pending:
        print("Nothing left to rate. Run with --calibrate.")
        return 0

    if stratified:
        print(f"Selecting {stratified} images spread across the severity range...")
        pending = stratified_sample(pending, detector, stratified)
        print(f"  selected {len(pending)}")

    panel_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[Path, Path, float, int]] = []
    for path in pending:
        image = cv2.imread(str(path))
        if image is None:
            print(f"  unreadable, skipping: {path}")
            continue
        result = detector.detect(image)
        panel_path = panel_dir / f"{path.stem}_review.jpg"
        cv2.imwrite(str(panel_path),
                    build_panel(image, result.overlay,
                                f"{path.name}  detector {result.severity_score:.3f}  "
                                f"contours {result.crack_count}"))
        written.append((path, panel_path, result.severity_score, result.crack_count))

    SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_PATH.write_text(json.dumps(
        [{"image": str(path.as_posix()), "panel": str(panel.as_posix()),
          "score": score, "contours": count}
         for path, panel, score, count in written], indent=2), encoding="utf-8")

    print(f"\nWrote {len(written)} review panels to {panel_dir}")
    if panels_only:
        print(f"Selection recorded in {SELECTION_PATH}. "
              f"Re-run without --panels-only to rate them.")
        return 0
    return prompt_for_ratings(written, labels, panel_dir)


def prompt_for_ratings(written: list[tuple[Path, Path, float, int]],
                       labels: dict[str, int], panel_dir: Path) -> int:
    """Take a 1-5 rating per panel, saving after each so a quit loses nothing."""
    print(f"Open {panel_dir} in an image viewer, then rate each below.")
    print("Original is on the left, detector overlay on the right.\n")
    for line in SCALE_HELP:
        print("  " + line)
    print("  s skip | q quit and save\n")

    for index, (path, panel_path, score, count) in enumerate(written, 1):
        prompt = (f"[{index}/{len(written)}] {panel_path.name}  "
                  f"(detector {score:.3f}, {count} contours) > ")
        while True:
            try:
                answer = input(prompt).strip().lower()
            except EOFError:
                answer = "q"
            if answer in ("q", "quit"):
                save_labels(labels)
                print(f"\nSaved {len(labels)} ratings to {LABEL_PATH}")
                return 0
            if answer in ("s", ""):
                break
            if answer in {"1", "2", "3", "4", "5"}:
                labels[str(path.as_posix())] = int(answer)
                save_labels(labels)
                break
            print("  enter 1-5, s to skip, or q to quit")

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
    parser.add_argument("--panel-dir", type=Path, default=Path("data/crack_review"),
                        help="Where review panels are written for viewing")
    parser.add_argument("--stratified", type=int, default=0,
                        help="Select this many images spread across the detector's "
                             "severity range instead of taking all of them")
    parser.add_argument("--panels-only", action="store_true",
                        help="Write the panels and exit without prompting. "
                             "Scoring the corpus to stratify takes minutes; this "
                             "keeps that out of the rating session.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seeds the stratified sampler so the same images are "
                             "chosen on a re-run")
    args = parser.parse_args()

    random.seed(args.seed)

    if args.calibrate:
        return run_calibration(args.treatable_max)

    directories = args.input_dir or [Path("data/crack_images"),
                                     Path("data/test_crack_detector")]
    return run_labelling(directories, args.panel_dir, stratified=args.stratified,
                         panels_only=args.panels_only)


if __name__ == "__main__":
    raise SystemExit(main())
