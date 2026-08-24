"""Pretrain AXUM's visual encoder on explicitly mapped isolated Ethiopic glyphs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OCR_IMG_SIZE
from src.ocr.model import GeezGlyphClassifier
from src.ocr.pipeline import resize_pad_image
from src.ocr.training_contract import charset_fingerprint, load_class_map


class IsolatedGlyphDataset(Dataset):
    """Mapped isolated-glyph images, split deterministically within each class."""

    def __init__(self, root: Path, class_map: dict[int, str], split: str, augment: bool):
        self.samples: list[tuple[Path, int]] = []
        self.characters = [class_map[index] for index in sorted(class_map)]
        class_position = {index: position for position, index in enumerate(sorted(class_map))}
        rng = random.Random(42)
        for index in sorted(class_map):
            folder = root / str(index)
            images = sorted(
                path for path in folder.glob("*")
                if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            rng.shuffle(images)
            boundary = int(len(images) * 0.85)
            selected = images[:boundary] if split == "train" else images[boundary:]
            self.samples.extend((path, class_position[index]) for path in selected)

        operations = [
            transforms.Lambda(
                lambda image: Image.fromarray(
                    resize_pad_image(np.array(image.convert("RGB")), OCR_IMG_SIZE)
                )
            ),
        ]
        if augment:
            operations.extend([
                transforms.RandomAffine(
                    degrees=8,
                    translate=(0.05, 0.05),
                    scale=(0.9, 1.1),
                    fill=255,
                ),
                transforms.ColorJitter(brightness=0.25, contrast=0.25),
            ])
        operations.extend([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        self.transform = transforms.Compose(operations)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image)
        return tensor, target


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = correct = total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * targets.size(0)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += targets.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretrain AXUM OCR encoder on isolated glyphs")
    parser.add_argument("--data-dir", type=Path, default=Path("data/geez_chars_clean/OCR_dataset/train"))
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/geez_glyph_encoder.pth"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    class_map = load_class_map(args.class_map)
    train_set = IsolatedGlyphDataset(args.data_dir, class_map, "train", augment=True)
    val_set = IsolatedGlyphDataset(args.data_dir, class_map, "val", augment=False)
    if not train_set or not val_set:
        raise RuntimeError("Mapped glyph dataset is empty; verify --data-dir and --class-map")

    device = torch.device(args.device)
    model = GeezGlyphClassifier(len(class_map)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=0)

    best_accuracy = -1.0
    history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })
        print(
            f"Epoch {epoch + 1:02d}/{args.epochs}: train={train_accuracy:.2%} "
            f"val={val_accuracy:.2%}"
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                "cnn_state_dict": {key: value.cpu() for key, value in model.cnn.state_dict().items()},
                "glyph_classifier_state_dict": {
                    key: value.cpu() for key, value in model.classifier.state_dict().items()
                },
                "class_map": {str(index): character for index, character in class_map.items()},
                "class_map_fingerprint": charset_fingerprint(train_set.characters),
                "image_size": list(OCR_IMG_SIZE),
                "best_val_accuracy": best_accuracy,
                "history": history,
            }, args.output)
    print(f"Saved encoder: {args.output} (best val accuracy={best_accuracy:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
