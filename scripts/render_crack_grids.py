"""
render_crack_grids.py — render images with a labelled coarse grid for rating.

WHY: a model is being designed around the claim that a human-level reader spots
cracks that the filters miss. That claim should be measured, not assumed. Pixel
masks cannot be produced by eye, but cell occupancy can: an 8x8 grid over a
512px image asks "does crack pass through this cell", which is answerable by
inspection and scorable against pixel ground truth collapsed to the same grid.

Cells are labelled A1..H8, column letter then row number, so an answer is a
plain list of cell names.

Usage:
    python scripts/render_crack_grids.py --dataset stone331 --count 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATASETS = {
    "stone331": (Path("data/crack_datasets/stone331/images"),
                 Path("data/crack_datasets/stone331/gt")),
    "mcs": (Path("data/crack_datasets/mcs/images"),
            Path("data/crack_datasets/mcs/masks")),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
COLUMNS = "ABCDEFGH"


def index_by_stem(directory: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(directory.rglob("*"))
            if p.suffix.lower() in IMAGE_SUFFIXES}


def draw_grid(image: np.ndarray, divisions: int, size: int) -> np.ndarray:
    """Upscale to a readable size and overlay a labelled grid."""
    canvas = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    step = size // divisions
    for index in range(1, divisions):
        cv2.line(canvas, (index * step, 0), (index * step, size), (0, 255, 255), 1)
        cv2.line(canvas, (0, index * step), (size, index * step), (0, 255, 255), 1)
    for col in range(divisions):
        for row in range(divisions):
            label = f"{COLUMNS[col]}{row + 1}"
            origin = (col * step + 4, row * step + 16)
            cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def truth_cells(mask: np.ndarray, divisions: int,
                min_pixels: int = 8) -> list[str]:
    """Cells the ground truth actually passes through."""
    resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    step = 512 // divisions
    cells = []
    for col in range(divisions):
        for row in range(divisions):
            tile = resized[row * step:(row + 1) * step,
                           col * step:(col + 1) * step]
            if np.count_nonzero(tile) >= min_pixels:
                cells.append(f"{COLUMNS[col]}{row + 1}")
    return sorted(cells)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="stone331", choices=sorted(DATASETS))
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--stride", type=int, default=37,
                        help="Take every Nth pair, to spread across the set")
    parser.add_argument("--divisions", type=int, default=8)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--out-dir", type=Path, default=Path("data/crack_grids"))
    args = parser.parse_args()

    image_dir, truth_dir = DATASETS[args.dataset]
    images = index_by_stem(image_dir)
    truths = index_by_stem(truth_dir)
    shared = sorted(set(images) & set(truths))
    picked = shared[::args.stride][:args.count]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    answers = {}
    for stem in picked:
        image = cv2.imread(str(images[stem]))
        mask = cv2.imread(str(truths[stem]), cv2.IMREAD_GRAYSCALE)
        out = args.out_dir / f"{args.dataset}_{stem}_grid.png"
        cv2.imwrite(str(out), draw_grid(image, args.divisions, args.size))
        answers[stem] = truth_cells(mask, args.divisions)
        print(f"{out}   truth occupies {len(answers[stem])}/"
              f"{args.divisions ** 2} cells")

    key = args.out_dir / f"{args.dataset}_truth.json"
    key.write_text(json.dumps(answers, indent=2), encoding="utf-8")
    print(f"\nwrote {len(picked)} grids and the answer key to {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
