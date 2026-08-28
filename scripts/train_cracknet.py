"""
train_cracknet.py — train the connectivity-aware crack segmenter.

Trains on MCS and Stone331 together. They annotate incompatibly (bodies ~21.9px
vs centrelines ~1.9px, a 23x difference established earlier), so rather than
merging them naively the body masks are SKELETONISED to centrelines. That makes
one consistent target definition and 577 training images instead of 246 or 331,
and the width information MCS uniquely carries is kept on a second head whose
loss is masked off for Stone331.

Full frames, not crops. The previous trainer sampled 256px windows from 512px
images to preserve native pixel scale for hairlines, which is correct on its own
terms but means the model never sees a whole crack -- and a controlled reading
experiment showed that seeing the whole crack is the entire advantage a human
reader has over the filters (cell-level F1 0.89 against 0.40).

Reported alongside pixel metrics is cell-level F1 on the same 8x8 grid the
reader was scored on, so the model's number is directly comparable to 0.89.

Usage:
    python scripts/train_cracknet.py --epochs 150 --batch-size 8 --amp
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.crack_detection.cracknet import CrackNet, CrackNetConfig, crack_loss

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

SOURCES = {
    "mcs": {"images": Path("data/crack_datasets/mcs/images"),
            "masks": Path("data/crack_datasets/mcs/masks"),
            "convention": "body"},
    "stone331": {"images": Path("data/crack_datasets/stone331/images"),
                 "masks": Path("data/crack_datasets/stone331/gt"),
                 "convention": "centreline"},
}

# GT-CrackSeg publishes DeepCrack, TUT, Khanhha and CrackTree in one layout:
# <name>/<Split>/img and <name>/<Split>/anno. Khanhha alone is ~11,200 images
# against our 577, and the binding constraint on this work has been data, not
# architecture. Anything dropped into data/crack_datasets in that shape is
# picked up automatically.
GT_SPLITS = ("Train", "Val", "Test")


def discover_gt_layout(root: Path) -> dict[str, dict]:
    """Find datasets stored in the GT-CrackSeg directory convention."""
    found = {}
    if not root.exists():
        return found
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for split in GT_SPLITS:
            images, masks = folder / split / "img", folder / split / "anno"
            if images.is_dir() and masks.is_dir():
                found[f"{folder.name}/{split}"] = {
                    "images": images, "masks": masks,
                    # Every one of these annotates the crack body; the
                    # centreline head takes their skeleton, as for MCS.
                    "convention": "body"}
    return found


def index_by_stem(directory: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(directory.rglob("*"))
            if p.suffix.lower() in IMAGE_SUFFIXES}


def collect_pairs() -> list[tuple[Path, Path, str]]:
    pairs = []
    catalogue = dict(SOURCES)
    catalogue.update(discover_gt_layout(Path("data/crack_datasets")))
    for name, spec in catalogue.items():
        images = index_by_stem(spec["images"])
        masks = index_by_stem(spec["masks"])
        shared = sorted(set(images) & set(masks))
        if shared:
            print(f"  {name:28s} {len(shared):>6,d} pairs  ({spec['convention']})")
        for stem in shared:
            pairs.append((images[stem], masks[stem], spec["convention"]))
    return pairs


class CrackPairs(Dataset):
    """
    Full-frame image with a centreline target and, where available, a width map.

    The centreline target is dilated to `target_width` pixels. A 1px target at
    512px is 0.1% positive, which starves the cross-entropy term; the dilation
    is undone at scoring time by matching with tolerance.
    """

    def __init__(self, pairs, size: int = 512, augment: bool = False,
                 target_width: int = 3, fixed_seed: int | None = None):
        self.pairs = pairs
        self.size = size
        self.augment = augment
        self.target_width = target_width
        self.fixed_seed = fixed_seed

    def __len__(self) -> int:
        return len(self.pairs)

    def _targets(self, mask: np.ndarray, convention: str):
        binary = (mask > 0).astype(np.uint8)
        if convention == "body":
            # Skeletonise AFTER resizing, so the centreline of the resized body
            # is taken rather than a resized centreline, which would fragment.
            centre = skeletonize(binary > 0).astype(np.uint8)
            # Half-width in pixels, normalised; the distance transform of the
            # body evaluated on its own skeleton is exactly that.
            distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            width = np.clip(distance / 32.0, 0.0, 1.0).astype(np.float32)
            valid = True
        else:
            centre = binary
            width = np.zeros_like(binary, dtype=np.float32)
            valid = False

        thick = cv2.dilate(centre, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.target_width, self.target_width)))
        return thick.astype(np.float32), width, valid

    def __getitem__(self, index: int):
        image_path, mask_path, convention = self.pairs[index]
        rng = (random if self.fixed_seed is None
               else random.Random(self.fixed_seed + index))

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        centre, width, valid = self._targets(mask, convention)

        if self.augment:
            if rng.random() < 0.5:
                image, centre, width = (np.fliplr(a).copy() for a in (image, centre, width))
            if rng.random() < 0.5:
                image, centre, width = (np.flipud(a).copy() for a in (image, centre, width))
            turns = rng.randint(0, 3)
            if turns:
                image, centre, width = (np.rot90(a, turns).copy()
                                        for a in (image, centre, width))
            # Lighting varies far more in the field than in either dataset.
            if rng.random() < 0.7:
                gain = 0.7 + 0.6 * rng.random()
                bias = rng.uniform(-25, 25)
                image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

        tensor = image.astype(np.float32) / 255.0
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        return (torch.from_numpy(tensor).permute(2, 0, 1),
                torch.from_numpy(centre).unsqueeze(0),
                torch.from_numpy(width).unsqueeze(0),
                torch.tensor(valid))


def cell_occupancy(mask: np.ndarray, divisions: int = 8,
                   min_pixels: int = 8) -> set[tuple[int, int]]:
    """Which grid cells a mask passes through, for comparison with the reader."""
    step = mask.shape[0] // divisions
    cells = set()
    for col in range(divisions):
        for row in range(divisions):
            tile = mask[row * step:(row + 1) * step, col * step:(col + 1) * step]
            if np.count_nonzero(tile) >= min_pixels:
                cells.add((col, row))
    return cells


@torch.no_grad()
def evaluate(model, loader, device, tolerance: int = 3) -> dict:
    """Pixel P/R/F1 swept over thresholds, plus cell-level F1."""
    model.eval()
    # [matched predicted, total predicted, matched truth, total truth]
    counts = {t: [0.0, 0.0, 0.0, 0.0] for t in THRESHOLDS}
    cell_scores = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * tolerance + 1, 2 * tolerance + 1))

    for images, centre, _width, _valid in loader:
        probability = torch.sigmoid(model(images.to(device))["centreline"]).cpu().numpy()
        truth = centre.numpy()
        for index in range(probability.shape[0]):
            truth_binary = (truth[index, 0] > 0.5).astype(np.uint8)
            slack = cv2.dilate(truth_binary, kernel)
            truth_total = float(np.count_nonzero(truth_binary))
            for t in THRESHOLDS:
                predicted = (probability[index, 0] > t).astype(np.uint8)
                counts[t][0] += float(np.count_nonzero(predicted & slack))
                counts[t][1] += float(np.count_nonzero(predicted))
                # Recall matched against the UNDILATED truth, so tolerance
                # cannot inflate it.
                counts[t][2] += float(np.count_nonzero(
                    cv2.dilate(predicted, kernel) & truth_binary))
                counts[t][3] += truth_total

            predicted = (probability[index, 0] > 0.5).astype(np.uint8)
            true_cells = cell_occupancy(truth_binary)
            pred_cells = cell_occupancy(predicted)
            hits = len(true_cells & pred_cells)
            precision = hits / len(pred_cells) if pred_cells else 0.0
            recall = hits / len(true_cells) if true_cells else 0.0
            cell_scores.append(2 * precision * recall / (precision + recall)
                               if (precision + recall) else 0.0)

    best_f1, best_threshold, best_pair = 0.0, 0.5, (0.0, 0.0)
    for t, (matched_pred, total_pred, matched_truth, total_truth) in counts.items():
        precision = matched_pred / total_pred if total_pred else 0.0
        recall = matched_truth / total_truth if total_truth else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        if f1 > best_f1:
            best_f1, best_threshold, best_pair = f1, t, (precision, recall)

    return {"best_f1": best_f1, "best_threshold": best_threshold,
            "precision": best_pair[0], "recall": best_pair[1],
            "cell_f1": float(np.mean(cell_scores)) if cell_scores else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1,
                        help="Pretrained features need a gentler step than a "
                             "randomly initialised decoder")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--cl-weight", type=float, default=0.5)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-coord-attention", action="store_true",
                        help="Ablate Coordinate Attention in the decoder")
    parser.add_argument("--no-deep-supervision", action="store_true",
                        help="Ablate the per-scale side outputs")
    parser.add_argument("--no-global-attention", action="store_true",
                        help="Ablate bottleneck self-attention")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", type=Path, default=Path("models/crack"))
    args = parser.parse_args()

    pairs = collect_pairs()
    if not pairs:
        print("No pairs. Run scripts/fetch_crack_datasets.py first.", file=sys.stderr)
        return 1
    random.Random(args.split_seed).shuffle(pairs)
    if args.limit:
        pairs = pairs[:args.limit]

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    split = max(1, int(len(pairs) * args.val_fraction))
    val_pairs, train_pairs = pairs[:split], pairs[split:]
    train_set = CrackPairs(train_pairs, args.size, augment=True)
    val_set = CrackPairs(val_pairs, args.size, augment=False, fixed_seed=1234)

    bodies = sum(1 for _, _, c in train_pairs if c == "body")
    print(f"{len(train_pairs)} train ({bodies} body, {len(train_pairs) - bodies} "
          f"centreline) | {len(val_pairs)} val | {args.size}px full frames")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              drop_last=len(train_set) > args.batch_size)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            num_workers=args.num_workers)

    device = torch.device(args.device)
    model = CrackNet(CrackNetConfig(
        pretrained=not args.no_pretrained,
        coord_attention=not args.no_coord_attention,
        deep_supervision=not args.no_deep_supervision,
        global_attention=not args.no_global_attention,
    )).to(device)
    print(f"parameters: {model.parameter_count():,}  device: {device}")

    positive = np.mean([float(train_set[i][1].mean())
                        for i in range(min(len(train_set), 40))])
    pos_weight = torch.tensor(
        [max(1.0, ((1 - positive) / max(positive, 1e-6)) ** 0.5)], device=device)
    print(f"positive pixels {100 * positive:.3f}%  pos_weight {float(pos_weight):.1f}")

    encoder_ids = {id(p) for module in (model.stem, model.layer1, model.layer2,
                                        model.layer3, model.layer4)
                   for p in module.parameters()}
    optimiser = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) in encoder_ids],
         "lr": args.lr * args.encoder_lr_scale},
        {"params": [p for p in model.parameters() if id(p) not in encoder_ids],
         "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=[args.lr * args.encoder_lr_scale, args.lr],
        total_steps=args.epochs * max(len(train_loader), 1), pct_start=0.1)

    run = f"cracknet_{args.tag}" if args.tag else "cracknet"
    args.out.mkdir(parents=True, exist_ok=True)
    best, history = 0.0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        started = time.time()
        for images, centre, width, valid in train_loader:
            images, centre = images.to(device), centre.to(device)
            width, valid = width.to(device), valid.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=args.amp and device.type == "cuda"):
                outputs = model(images)
                loss, _terms = crack_loss(outputs, centre, width, valid,
                                          pos_weight, args.cl_weight)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            running += float(loss.detach())

        metrics = evaluate(model, val_loader, device)
        metrics.update(epoch=epoch, loss=running / max(len(train_loader), 1))
        history.append(metrics)
        print(f"  epoch {epoch:>3} | loss {metrics['loss']:7.3f} "
              f"| pixel F1 {metrics['best_f1']:.3f} @{metrics['best_threshold']:.1f} "
              f"| cell F1 {metrics['cell_f1']:.3f} "
              f"| {time.time() - started:5.1f}s")

        if metrics["cell_f1"] > best:
            best = metrics["cell_f1"]
            torch.save({"state_dict": model.state_dict(),
                        "config": vars(model.config),
                        "cell_f1": best,
                        "threshold": metrics["best_threshold"]},
                       args.out / f"{run}.pth")

    log = Path("logs/crack_segmenter") / f"{run}_training.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(
        {"epochs": args.epochs, "images": len(pairs),
         "parameters": model.parameter_count(), "best_cell_f1": best,
         "reader_cell_f1": 0.89, "filter_cell_f1": 0.40,
         "history": history}, indent=2), encoding="utf-8")
    print(f"\nbest cell F1 {best:.3f} (reader 0.89, filter pipeline 0.40)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
