"""Probe: at which stage does the wall crack get lost?"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.crack_detection.detector import CrackDetector

path = next(Path("data/crack_images/test").glob("Detail_of_vertical*"))
full = cv2.imread(str(path))
height, width = full.shape[:2]
wall = full[int(0.30 * height):int(0.82 * height), int(0.18 * width):width]

detector = CrackDetector()
gray = detector._to_gray(wall)
working = detector._to_working_scale(gray)
flattened = detector._flatten_illumination(working)
response = detector._ridge_response(flattened)
seeds = response > detector.ridge_threshold
dark = flattened < detector.darkness_cut
grown = detector._grow_from_ridges(response, flattened)
bridged = detector._bridge_gaps(grown)
final, boxes, _ = detector._keep_crack_shapes(bridged)

print(f"working {working.shape}")
print(f"flattened   min {flattened.min():6.1f}  median {np.median(flattened):6.1f}  "
      f"max {flattened.max():6.1f}")
print(f"response    max {response.max():.4f}  p99.9 {np.percentile(response, 99.9):.4f}")
print(f"seeds       {100 * seeds.mean():6.3f}% of pixels")
print(f"dark        {100 * dark.mean():6.3f}%")
print(f"grown       {100 * (grown > 0).mean():6.3f}%")
print(f"after gate  {100 * (final > 0).mean():6.3f}%  ({len(boxes)} regions)")

# Sample a column that crosses the crack, to see what each stage says there.
column = np.argmin(cv2.GaussianBlur(working, (0, 0), 3).mean(axis=0))
print(f"\ndarkest column x={column}; values down that column every 40px")
print(f"{'y':>5s} {'gray':>6s} {'flat':>7s} {'ridge':>8s} {'seed':>5s} "
      f"{'dark':>5s} {'grown':>6s} {'final':>6s}")
for y in range(0, working.shape[0], 40):
    print(f"{y:>5d} {working[y, column]:>6d} {flattened[y, column]:>7.1f} "
          f"{response[y, column]:>8.4f} {str(bool(seeds[y, column])):>5s} "
          f"{str(bool(dark[y, column])):>5s} {str(grown[y, column] > 0):>6s} "
          f"{str(final[y, column] > 0):>6s}")

stages = [("working", cv2.cvtColor(working, cv2.COLOR_GRAY2BGR)),
          ("flattened", cv2.cvtColor(flattened.astype(np.uint8), cv2.COLOR_GRAY2BGR)),
          ("ridge response x8", cv2.applyColorMap(
              np.clip(response * 8 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)),
          ("seeds", cv2.cvtColor(seeds.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)),
          ("dark", cv2.cvtColor(dark.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)),
          ("grown", cv2.cvtColor(grown, cv2.COLOR_GRAY2BGR)),
          ("after shape gate", cv2.cvtColor(final, cv2.COLOR_GRAY2BGR))]
tiles = []
for label, tile in stages:
    bar = np.zeros((26, tile.shape[1], 3), np.uint8)
    cv2.putText(bar, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1)
    tiles.append(np.vstack([tile, bar]))
out = Path("data/crack_review/comparison/_stages.jpg")
cv2.imwrite(str(out), np.hstack(tiles[:4]))
cv2.imwrite(str(out.with_name("_stages2.jpg")), np.hstack(tiles[4:]))
print(f"\nwrote {out} and _stages2.jpg")
