"""Probe: joint calibration of the texture multiple across datasets and photos."""
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

logger.remove()  # the detector logs INFO per call, which floods a sweep

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.crack_detection.detector import CrackDetector
from scripts.experiment_crack_methods import index_by_stem, score

PHOTOS = sorted(Path("data/crack_images/test").glob("*.jpg"))
MULTIPLES = (0.0, 3.0, 4.0)


def dataset_score(detector, folder, truth_folder, limit=15):
    images = index_by_stem(Path(folder))
    truths = index_by_stem(Path(truth_folder))
    stems = sorted(set(images) & set(truths))[:limit]
    scores = []
    for stem in stems:
        image = cv2.imread(str(images[stem]))
        truth = cv2.imread(str(truths[stem]), cv2.IMREAD_GRAYSCALE)
        result = detector.detect(image)
        truth = cv2.resize(truth, (result.mask.shape[1], result.mask.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        scores.append(score((result.mask > 0).astype(np.uint8), truth))
    return (float(np.mean([s[0] for s in scores])),
            float(np.mean([s[1] for s in scores])),
            float(np.mean([s[2] for s in scores])))


blank = np.full((400, 600, 3), 160, np.uint8)
print(f"{'mult':>5s} {'width':>6s} {'MCS P':>7s} {'MCS R':>7s} {'MCS F1':>7s} "
      f"{'S331 F1':>8s} {'blank':>6s}  photos: regions found")
print("-" * 100)
for width in (40.0, 80.0):
    for multiple in MULTIPLES:
        detector = CrackDetector(texture_multiple=multiple, max_mean_width=width)
        precision, recall, f1 = dataset_score(
            detector, "data/crack_datasets/mcs/images",
            "data/crack_datasets/mcs/masks")
        _, _, stone_f1 = dataset_score(
            detector, "data/crack_datasets/stone331/images",
            "data/crack_datasets/stone331/gt")
        blank_count = detector.detect(blank).crack_count
        found = []
        for path in PHOTOS:
            image = cv2.imread(str(path))
            scale = 700 / image.shape[1]
            small = cv2.resize(image, (700, int(image.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)
            result = detector.detect(small)
            found.append(f"{result.crack_count:>3d}/{100 * (result.mask > 0).mean():4.1f}%")
        print(f"{multiple:>5.1f} {width:>6.0f} {precision:>7.3f} {recall:>7.3f} "
              f"{f1:>7.3f} {stone_f1:>8.3f} {blank_count:>6d}  " + " ".join(found))

print("\nphoto order:")
for path in PHOTOS:
    print(f"  {path.name[:70]}")
