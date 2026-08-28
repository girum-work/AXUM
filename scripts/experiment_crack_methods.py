"""
experiment_crack_methods.py — compare crack response maps against ground truth.

WHY: the deployed detector marks a fixed 2% of pixels on every image and cannot
see a crack wider than its 15px black-hat kernel, so it fires on lichen and
grain while missing the obvious fissure. Replacing it is only defensible if the
replacement is measured on the same data, so this scores several candidate
response maps on Stone331's pixel-level ground truth.

Each method returns a float response map, higher meaning more crack-like. Two
numbers are reported per method:

    best percentile F1   how well the map RANKS crack above non-crack, which is
                         the map's intrinsic quality
    fixed threshold F1   what a deployable absolute cut-off achieves, which is
                         what lets a crack-free rock return an empty mask

The deployed detector only has the first kind: a percentile keeps 2% of pixels
whether or not any crack is present.

Usage:
    python scripts/experiment_crack_methods.py --limit 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.crack_detection.detector import CrackDetector

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PERCENTILES = (99.99, 99.95, 99.9, 99.8, 99.5, 99.0, 98.0, 95.0)
FIXED_FRACTIONS = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6)
DATASETS = {
    "stone331": (Path("data/crack_datasets/stone331/images"),
                 Path("data/crack_datasets/stone331/gt")),
    "mcs": (Path("data/crack_datasets/mcs/images"),
            Path("data/crack_datasets/mcs/masks")),
}


def index_by_stem(directory: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(directory.rglob("*"))
            if p.suffix.lower() in IMAGE_SUFFIXES}


def prepare(image: np.ndarray, long_edge: int) -> np.ndarray:
    """
    Grayscale, scale-normalised, illumination-flattened.

    Kernel sizes are in pixels, so an unnormalised 12MP photo and a 0.3MP one
    are effectively processed at different physical scales. Fixing the long
    edge makes one kernel mean one thing everywhere.
    """
    gray = (image if image.ndim == 2
            else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    height, width = gray.shape
    if max(height, width) != long_edge:
        scale = long_edge / max(height, width)
        gray = cv2.resize(gray, (max(1, int(width * scale)),
                                 max(1, int(height * scale))),
                          interpolation=cv2.INTER_AREA)
    # Divide out slow shading so a dark corner is not read as damage.
    background = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), long_edge / 16)
    flat = gray.astype(np.float32) / np.maximum(background, 1.0)
    return np.clip(flat * 128.0, 0, 255)


def hessian_valley(gray: np.ndarray, sigmas=(1.0, 2.0, 4.0, 8.0)) -> np.ndarray:
    """
    Multi-scale dark-ridge (valley) response from Hessian eigenvalues.

    A crack is a dark curvilinear valley: curvature across it is large and
    positive, curvature along it is near zero. Black-hat cannot make that
    distinction -- it responds to any dark structure, so a compact grain of
    lichen scores as high as a fissure. The eigenvalue ratio is exactly the
    measurement that separates the two.

    Scale normalisation (sigma^2) is what makes responses from different sigmas
    comparable, so one filter bank covers hairlines and wide fissures at once.
    """
    response = np.zeros_like(gray, dtype=np.float32)
    for sigma in sigmas:
        smooth = cv2.GaussianBlur(gray, (0, 0), sigma)
        gxx = sigma ** 2 * cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3)
        gyy = sigma ** 2 * cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3)
        gxy = sigma ** 2 * cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3)

        spread = np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2)
        big = 0.5 * (gxx + gyy + spread)
        small = 0.5 * (gxx + gyy - spread)
        # |small| <= |big| by construction; order them by magnitude.
        lam_low = np.where(np.abs(big) < np.abs(small), big, small)
        lam_high = np.where(np.abs(big) < np.abs(small), small, big)

        # Dark line on a lighter surface: lam_high must be positive.
        blobness = np.abs(lam_low) / (np.abs(lam_high) + 1e-6)
        structure = np.sqrt(lam_low ** 2 + lam_high ** 2)
        # Frangi's c: half the strongest structure in this image. A fixed
        # constant is wrong because the response scales with contrast, which
        # differs between a lichen-covered boulder and a clean marble slab.
        half_max = 0.5 * float(structure.max()) or 1.0
        scale_response = (np.exp(-(blobness ** 2) / 0.5)
                          * (1.0 - np.exp(-(structure ** 2) / (2 * half_max ** 2))))
        scale_response[lam_high <= 0] = 0.0
        response = np.maximum(response, scale_response)
    return response


def blackhat_multiscale(gray: np.ndarray,
                        widths=(7, 15, 31, 63)) -> np.ndarray:
    """Current approach, but over a kernel bank so wide cracks also respond."""
    source = gray.astype(np.uint8)
    response = np.zeros_like(source, dtype=np.float32)
    for width in widths:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, width))
        response = np.maximum(
            response,
            cv2.morphologyEx(source, cv2.MORPH_BLACKHAT, kernel).astype(np.float32))
    return response / 255.0


def blackhat_oriented(gray: np.ndarray, length: int = 31,
                      angles=range(0, 180, 15)) -> np.ndarray:
    """
    Black-hat with long thin structuring elements, maximised over orientation.

    A line element only closes a structure that is elongated across it, so an
    isotropic speck of lichen is suppressed at every angle while a crack
    responds strongly at one. This is the cheapest way to buy anisotropy
    without computing a Hessian.
    """
    source = gray.astype(np.uint8)
    response = np.zeros_like(source, dtype=np.float32)
    base = np.zeros((length, length), dtype=np.uint8)
    base[length // 2, :] = 1
    centre = (length / 2 - 0.5, length / 2 - 0.5)
    for angle in angles:
        rotation = cv2.getRotationMatrix2D(centre, float(angle), 1.0)
        kernel = cv2.warpAffine(base, rotation, (length, length),
                                flags=cv2.INTER_NEAREST)
        if not kernel.any():
            continue
        response = np.maximum(
            response,
            cv2.morphologyEx(source, cv2.MORPH_BLACKHAT, kernel).astype(np.float32))
    return response / 255.0


def skimage_ridge(name: str, sigmas=(1, 2, 4, 8)):
    """Reference implementations, to check the hand-rolled Hessian is not worse."""
    from skimage.filters import frangi, meijering, sato
    function = {"frangi": frangi, "sato": sato, "meijering": meijering}[name]

    def apply(gray: np.ndarray) -> np.ndarray:
        out = function(gray / 255.0, sigmas=sigmas, black_ridges=True)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(out.max())
        return (out / peak).astype(np.float32) if peak > 0 else out.astype(np.float32)

    return apply


def current_detector_mask(detector: CrackDetector, image: np.ndarray,
                          long_edge: int) -> np.ndarray:
    """The deployed detector's own mask, at the same working resolution."""
    gray = (image if image.ndim == 2
            else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    height, width = gray.shape
    scale = long_edge / max(height, width)
    resized = cv2.resize(image, (max(1, int(width * scale)),
                                 max(1, int(height * scale))),
                         interpolation=cv2.INTER_AREA)
    return (detector.detect(resized).mask > 0).astype(np.uint8)


def elongation_gate(mask: np.ndarray, min_elongation: float = 3.0,
                    min_extent: int = 12) -> np.ndarray:
    """
    Drop components that are not long and thin.

    Ranking pixels well is not enough: lichen and grain survive any threshold
    because they genuinely are dark. They do not survive a shape test.
    Elongation is the ratio of the component's principal axes, taken from the
    eigenvalues of its pixel covariance.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    kept = np.zeros_like(mask)
    for index in range(1, count):
        area = stats[index, cv2.CC_STAT_AREA]
        if area < 6:
            continue
        ys, xs = np.nonzero(labels == index)
        coords = np.stack([xs, ys]).astype(np.float32)
        covariance = np.cov(coords)
        if covariance.ndim < 2:
            continue
        eigenvalues = np.linalg.eigvalsh(covariance)
        major = float(np.sqrt(max(eigenvalues[-1], 1e-6)))
        minor = float(np.sqrt(max(eigenvalues[0], 1e-6)))
        if major * 4 < min_extent or major / max(minor, 1e-6) < min_elongation:
            continue
        kept[labels == index] = 1
    return kept


def score(predicted: np.ndarray, truth: np.ndarray,
          tolerance: int = 2) -> tuple[float, float, float]:
    """Precision/recall/F1 with slack around hand-traced centrelines."""
    truth_binary = (truth > 0).astype(np.uint8)
    predicted = predicted.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * tolerance + 1, 2 * tolerance + 1))
    truth_slack = cv2.dilate(truth_binary, kernel)

    hits = float(np.count_nonzero(predicted & truth_slack))
    predicted_total = float(np.count_nonzero(predicted))
    truth_total = float(np.count_nonzero(truth_binary))
    recall_hits = float(np.count_nonzero(
        cv2.dilate(predicted, np.ones((3, 3), np.uint8)) & truth_binary))

    precision = hits / predicted_total if predicted_total else 0.0
    recall = recall_hits / truth_total if truth_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return precision, recall, f1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="stone331", choices=sorted(DATASETS))
    parser.add_argument("--images", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--long-edge", type=int, default=512)
    parser.add_argument("--gate", action="store_true",
                        help="Apply the elongation gate to every method")
    parser.add_argument("--report", type=Path,
                        default=Path("logs/crack_method_experiment.json"))
    args = parser.parse_args()

    default_images, default_truth = DATASETS[args.dataset]
    args.images = args.images or default_images
    args.ground_truth = args.ground_truth or default_truth

    images = index_by_stem(args.images)
    truths = index_by_stem(args.ground_truth)
    shared = sorted(set(images) & set(truths))[:args.limit or None]
    if not shared:
        print(f"No pairs under {args.images}", file=sys.stderr)
        return 1

    detector = CrackDetector()
    methods = {
        "blackhat_multiscale": blackhat_multiscale,
        "blackhat_oriented": blackhat_oriented,
        "hessian_fine": lambda g: hessian_valley(g, sigmas=(0.5, 1.0, 2.0)),
        "hessian_coarse": hessian_valley,
        "frangi": skimage_ridge("frangi"),
        "sato_fine": skimage_ridge("sato", sigmas=(0.5, 1, 2)),
        "sato": skimage_ridge("sato"),
        "meijering_fine": skimage_ridge("meijering", sigmas=(0.5, 1, 2)),
        "meijering": skimage_ridge("meijering"),
    }

    print(f"{len(shared)} {args.dataset} pairs at {args.long_edge}px long edge"
          f"{' with elongation gate' if args.gate else ''}\n")

    totals: dict[str, dict[str, list[float]]] = {}
    clean_rates: dict[str, dict[str, list[float]]] = {}
    current_scores: list[tuple[float, float, float]] = []
    current_clean: list[float] = []
    timings: dict[str, float] = {name: 0.0 for name in methods}

    for stem in shared:
        image = cv2.imread(str(images[stem]))
        truth = cv2.imread(str(truths[stem]), cv2.IMREAD_GRAYSCALE)
        if image is None or truth is None:
            continue
        if truth.shape[:2] != image.shape[:2]:
            truth = cv2.resize(truth, (image.shape[1], image.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        scale = args.long_edge / max(truth.shape)
        truth_small = cv2.resize(truth, (max(1, int(truth.shape[1] * scale)),
                                         max(1, int(truth.shape[0] * scale))),
                                 interpolation=cv2.INTER_NEAREST)
        gray = prepare(image, args.long_edge)

        # Pixels well clear of any annotated crack. Anything marked here is a
        # false positive by construction -- this is the confetti measurement.
        far = cv2.dilate((truth_small > 0).astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
        clean = far == 0

        current_mask = current_detector_mask(detector, image, args.long_edge)
        current_scores.append(score(current_mask, truth_small))
        current_clean.append(float(current_mask[clean].mean()))

        for name, function in methods.items():
            started = time.time()
            response = function(gray)
            timings[name] += time.time() - started
            bucket = totals.setdefault(name, {})
            clean_bucket = clean_rates.setdefault(name, {})
            for percentile in PERCENTILES:
                cut = float(np.percentile(response, percentile))
                mask = (response > cut).astype(np.uint8)
                if args.gate:
                    mask = elongation_gate(mask)
                key = f"p{percentile}"
                bucket.setdefault(key, []).append(score(mask, truth_small)[2])
                clean_bucket.setdefault(key, []).append(float(mask[clean].mean()))
            peak = float(response.max()) or 1.0
            for fraction in FIXED_FRACTIONS:
                mask = (response > fraction * peak).astype(np.uint8)
                if args.gate:
                    mask = elongation_gate(mask)
                key = f"f{fraction}"
                bucket.setdefault(key, []).append(score(mask, truth_small)[2])
                clean_bucket.setdefault(key, []).append(float(mask[clean].mean()))

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    print(f"{'method':22s} {'best percentile':>22s} {'best fixed':>22s} "
          f"{'FP on clean':>12s} {'s/img':>7s}")
    print("-" * 90)
    print(f"{'CURRENT (deployed)':22s} "
          f"{f'F1 {mean([s[2] for s in current_scores]):.3f}':>22s} "
          f"{'n/a - percentile only':>22s} {mean(current_clean):11.2%} ")

    summary: dict[str, dict] = {
        "current": {"f1": mean([s[2] for s in current_scores]),
                    "precision": mean([s[0] for s in current_scores]),
                    "recall": mean([s[1] for s in current_scores]),
                    "false_positive_rate_on_clean": mean(current_clean)}}
    for name in methods:
        buckets = totals[name]
        best_p = max((k for k in buckets if k.startswith("p")),
                     key=lambda k: mean(buckets[k]))
        best_f = max((k for k in buckets if k.startswith("f")),
                     key=lambda k: mean(buckets[k]))
        summary[name] = {"best_percentile": best_p,
                         "percentile_f1": mean(buckets[best_p]),
                         "best_fixed": best_f, "fixed_f1": mean(buckets[best_f]),
                         "false_positive_rate_on_clean": mean(clean_rates[name][best_f]),
                         "seconds_per_image": timings[name] / len(shared)}
        print(f"{name:22s} {f'F1 {mean(buckets[best_p]):.3f} @ {best_p[1:]}':>22s} "
              f"{f'F1 {mean(buckets[best_f]):.3f} @ {best_f[1:]}':>22s} "
              f"{mean(clean_rates[name][best_f]):11.2%} "
              f"{timings[name] / len(shared):7.2f}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        {"dataset": args.dataset, "images": len(shared),
         "long_edge": args.long_edge, "gate": args.gate,
         "summary": summary}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
