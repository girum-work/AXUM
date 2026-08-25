"""
train_crack_segmenter.py — train the crack U-Net on one annotation convention.

Two labelled sets are available and they annotate differently: MCS marks the
crack BODY (21.9px wide on 256px frames), Stone331 traces a thin CENTRELINE
(1.9px on 512px). That is a 23x difference, so they are trained separately and
compared rather than merged.

    --dataset mcs        bodies, 246 marble images, CC BY 4.0
    --dataset stone331   centrelines, 331 stone images, no licence declared

Which convention is better is an open question this script exists to answer.
Centrelines keep the OpenCV baseline (F1 0.111) comparable; bodies carry width,
which is what severity actually depends on. Both models are therefore scored on
mask F1 AND on how well predicted crack area ranks true crack area, since
ranking damage is what the conservation pipeline consumes.

MCS masks are paletted PNGs. Read via cv2.IMREAD_GRAYSCALE and threshold > 0;
an IMREAD_UNCHANGED read returns crack only in channel 2, so taking channel 0
yields empty masks and trains on nothing.

Usage:
    python scripts/train_crack_segmenter.py --dataset mcs --epochs 80
    python scripts/train_crack_segmenter.py --dataset stone331 --epochs 80
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.crack_detection.segmenter import (
    CrackUNet,
    SegmenterConfig,
    combined_loss,
    save_segmenter,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

DATASETS = {
    "mcs": {
        "root": Path("data/crack_datasets/mcs"),
        "images": "images", "masks": "masks",
        "convention": "body",
    },
    "stone331": {
        "root": Path("data/crack_datasets/stone331"),
        "images": "images", "masks": "gt",
        "convention": "centreline",
    },
}


def index_by_stem(directory: Path) -> dict[str, Path]:
    """Map filename stem to path so image/mask pairing survives extensions."""
    return {p.stem: p for p in sorted(directory.rglob("*"))
            if p.suffix.lower() in IMAGE_SUFFIXES}


class CrackDataset(Dataset):
    """
    Image/mask pairs at a fixed square size, with light augmentation.

    Two sampling modes, because the datasets need different treatment.

    Resizing suits MCS, whose cracks are ~22px wide. It ruins Stone331: its
    centrelines are 1.9px, and downscaling 512 -> 256 with nearest neighbour
    shatters one connected crack into ~29 fragments. Positive rate and width
    survive the resize, connectivity does not, so the model is taught scattered
    dots rather than a continuous curve.

    Cropping keeps native pixel scale, so a crack stays connected, and doubles
    as augmentation: 331 images yield far more distinct 256px views.
    """

    def __init__(self, pairs: list[tuple[Path, Path]], size: int,
                 augment: bool = False, crop: bool = False,
                 min_positive: float = 0.0):
        self.pairs = pairs
        self.size = size
        self.augment = augment
        self.crop = crop
        self.min_positive = min_positive

    def __len__(self) -> int:
        return len(self.pairs)

    def _random_crop(self, image: np.ndarray, mask: np.ndarray,
                     attempts: int = 12) -> tuple[np.ndarray, np.ndarray]:
        """
        Take a native-scale window, preferring one that contains crack.

        Cracks cover ~0.1% of Stone331, so uniform crops are almost all
        background and the model learns to answer "no crack" every time.
        """
        height, width = mask.shape[:2]
        if height <= self.size or width <= self.size:
            return image, mask

        best = None
        for _ in range(attempts):
            y = random.randint(0, height - self.size)
            x = random.randint(0, width - self.size)
            mask_crop = mask[y:y + self.size, x:x + self.size]
            positive = float((mask_crop > 0).mean())
            if best is None or positive > best[0]:
                best = (positive, y, x)
            if positive >= self.min_positive:
                break
        _positive, y, x = best
        return (image[y:y + self.size, x:x + self.size],
                mask[y:y + self.size, x:x + self.size])

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if self.crop:
            # Match the image to the mask's own scale before cropping, so a
            # crop is a true native-resolution window rather than a rescale.
            if image.shape[:2] != mask.shape[:2]:
                image = cv2.resize(image, (mask.shape[1], mask.shape[0]),
                                   interpolation=cv2.INTER_AREA)
            image, mask = self._random_crop(image, mask)
            if image.shape[0] != self.size or image.shape[1] != self.size:
                image = cv2.resize(image, (self.size, self.size),
                                   interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (self.size, self.size),
                                  interpolation=cv2.INTER_NEAREST)
        else:
            image = cv2.resize(image, (self.size, self.size),
                               interpolation=cv2.INTER_AREA)
            # Nearest keeps the mask binary; anything smoother invents grey edges.
            mask = cv2.resize(mask, (self.size, self.size),
                              interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if random.random() < 0.5:
                image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
            if random.random() < 0.5:
                image, mask = np.flipud(image).copy(), np.flipud(mask).copy()
            turns = random.randint(0, 3)
            if turns:
                image = np.rot90(image, turns).copy()
                mask = np.rot90(mask, turns).copy()

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5) -> dict:
    """Mask F1/IoU, plus how well predicted area ranks true area."""
    model.eval()
    tp = fp = fn = 0.0
    predicted_areas: list[float] = []
    true_areas: list[float] = []

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        probability = torch.sigmoid(model(images))
        predicted = (probability > threshold).float()

        tp += float((predicted * masks).sum())
        fp += float((predicted * (1 - masks)).sum())
        fn += float(((1 - predicted) * masks).sum())

        for i in range(images.size(0)):
            predicted_areas.append(float(predicted[i].mean()))
            true_areas.append(float(masks[i].mean()))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou,
            "area_rank_corr": spearman(true_areas, predicted_areas)}


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation without pulling in scipy."""
    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = shared
            i = j + 1
        return ranks

    if len(a) < 2:
        return 0.0
    ra, rb = rank(a), rank(b)
    n = len(a)
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    var_a = sum((x - mean_a) ** 2 for x in ra)
    var_b = sum((y - mean_b) ** 2 for y in rb)
    if var_a == 0 or var_b == 0:
        return 0.0
    return cov / (var_a * var_b) ** 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--size", type=int, default=256,
                        help="Square training resolution; must divide by 2**depth")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--pos-weight", type=float, default=0.0,
                        help="0 = derive from the data's own class balance")
    parser.add_argument("--crop", action="store_true",
                        help="Sample native-scale windows instead of rescaling. "
                             "Required for Stone331: downscaling breaks its "
                             "1.9px centrelines into disconnected fragments.")
    parser.add_argument("--min-positive", type=float, default=0.0005,
                        help="Crop sampler retries until a window holds at least "
                             "this fraction of crack pixels")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("models/crack"))
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    spec = DATASETS[args.dataset]
    root = spec["root"]
    images = index_by_stem(root / spec["images"])
    masks = index_by_stem(root / spec["masks"])
    shared = sorted(set(images) & set(masks))
    if not shared:
        print(f"No pairs under {root}. Run scripts/fetch_crack_datasets.py first.",
              file=sys.stderr)
        return 1

    random.shuffle(shared)
    split = max(1, int(len(shared) * args.val_fraction))
    val_stems, train_stems = shared[:split], shared[split:]
    to_pairs = lambda stems: [(images[s], masks[s]) for s in stems]

    train_set = CrackDataset(to_pairs(train_stems), args.size, augment=True,
                             crop=args.crop, min_positive=args.min_positive)
    val_set = CrackDataset(to_pairs(val_stems), args.size, augment=False,
                           crop=args.crop, min_positive=args.min_positive)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=len(train_set) > args.batch_size)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            num_workers=args.num_workers)

    positive_rate = float(np.mean([
        float(train_set[i][1].mean()) for i in range(min(len(train_set), 60))
    ]))
    # The textbook balanced weight, (1 - p) / p, is degenerate at these rates.
    # Measured on Stone331 (p = 0.26%, balanced = 387), 6 epochs:
    #     weight 387  P 0.020  R 0.923  F1 0.040   predicts crack everywhere
    #     weight   1  P 0.000  R 0.000  F1 0.000   predicts nothing
    #     weight  20  P 0.103  R 0.263  F1 0.148   beats OpenCV's 0.111
    # 20 is sqrt(387), the geometric mean of the two failures, so the square
    # root is the default. Dice already absorbs imbalance; a full balanced
    # weight makes BCE fight it.
    balanced = (1 - positive_rate) / max(positive_rate, 1e-6)
    pos_weight = (torch.tensor([args.pos_weight]) if args.pos_weight > 0
                  else torch.tensor([max(1.0, balanced ** 0.5)]))

    device = torch.device(args.device)
    model = CrackUNet(SegmenterConfig(base_channels=args.base_channels)).to(device)
    print(f"{args.dataset} ({spec['convention']}): {len(train_stems)} train, "
          f"{len(val_stems)} val | positive pixels {100 * positive_rate:.3f}% "
          f"| pos_weight {float(pos_weight):.1f}")
    print(f"parameters: {model.parameter_count():,}  device: {device}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr,
        total_steps=args.epochs * max(len(train_loader), 1), pct_start=0.1)
    pos_weight = pos_weight.to(device)

    best = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        started = time.time()
        for batch_images, batch_masks in train_loader:
            batch_images = batch_images.to(device)
            batch_masks = batch_masks.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=args.amp and device.type == "cuda"):
                logits = model(batch_images)
                loss = combined_loss(logits, batch_masks, pos_weight=pos_weight)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            running += float(loss.detach())

        metrics = evaluate(model, val_loader, device)
        metrics.update(epoch=epoch, loss=running / max(len(train_loader), 1))
        history.append(metrics)
        print(f"  epoch {epoch:>3} | loss {metrics['loss']:6.3f} "
              f"| F1 {metrics['f1']:6.3f} | IoU {metrics['iou']:6.3f} "
              f"| P {metrics['precision']:.3f} R {metrics['recall']:.3f} "
              f"| areaCorr {metrics['area_rank_corr']:+.3f} "
              f"| {time.time() - started:5.1f}s")

        if metrics["f1"] > best:
            best = metrics["f1"]
            save_segmenter(model, args.out / f"crack_{args.dataset}.pth", {
                "dataset": args.dataset,
                "convention": spec["convention"],
                "size": args.size,
                "best_f1": best,
                "positive_rate": positive_rate,
            })

    log_path = Path("logs/crack_segmenter") / f"{args.dataset}_training.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "dataset": args.dataset,
        "convention": spec["convention"],
        "parameters": model.parameter_count(),
        "best_f1": best,
        "opencv_baseline_f1": 0.111,
        "history": history,
    }, indent=2), encoding="utf-8")

    print(f"\nbest F1 {best:.3f} (OpenCV detector scores 0.111 on Stone331)")
    print(f"saved to {args.out / f'crack_{args.dataset}.pth'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
