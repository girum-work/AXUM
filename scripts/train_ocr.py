# scripts/train_ocr.py
"""
Entry point for Ge'ez OCR model training.

WHAT: Sets up class-balanced sampling and delegates to train_ocr_model().
WHY: Keeps training orchestration in scripts/ while core logic lives in pipeline.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

from torch.utils.data import WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr.pipeline import train_ocr_model


def create_weighted_sampler(dataset) -> WeightedRandomSampler:
    """
    Creates a sampler that oversamples rare characters and undersamples common ones.

    WHAT: Assigns each training sample a weight inversely proportional to its
    primary character frequency so rare Ge'ez glyphs appear more often per epoch.
    WHY: Class imbalance is the single highest-impact fix for Ethiopic OCR —
    common syllables like ሰ dominate loss without rebalancing.

    Args:
        dataset: HHDEthiopicDataset (or any Dataset yielding
                 (_, label_tensor, _) tuples with CTC index labels)

    Returns:
        WeightedRandomSampler configured for replacement oversampling.
    """
    label_counts = Counter()
    for _, label_tensor, _ in dataset:
        for idx in label_tensor.tolist():
            label_counts[idx] += 1

    total = sum(label_counts.values())
    class_weights = {
        cls: total / (len(label_counts) * count)
        for cls, count in label_counts.items()
    }

    sample_weights = []
    for _, label_tensor, _ in dataset:
        first_char = label_tensor[0].item() if len(label_tensor) > 0 else 0
        weight = class_weights.get(first_char, 1.0)
        sample_weights.append(weight)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    most_common = label_counts.most_common(5)
    least_common = label_counts.most_common()[:-6:-1]
    print("\nClass balance report:")
    print(f"  Most common:  {most_common}")
    print(f"  Least common: {least_common}")
    print(f"  Total classes with data: {len(label_counts)}")

    return sampler


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for OCR training runs."""
    parser = argparse.ArgumentParser(description="Train Ge'ez OCR model")
    parser.add_argument("--data-dir", default="data/geez_characters")
    parser.add_argument("--save-path", default="models/geez_ocr.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--weighted-sampler",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--beam-decode",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--stone-augment",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--adaptive-binarize",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Starting Ge'ez OCR model training...")
    print(f"Epochs: {args.epochs}, batch: {args.batch_size}, lr: {args.lr}")
    print(f"Best model saves to {args.save_path}\n")

    train_ocr_model(
        data_dir=args.data_dir,
        save_path=Path(args.save_path),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_weighted_sampler=args.weighted_sampler,
        use_beam_val_decode=args.beam_decode,
        use_stone_augment=args.stone_augment,
        use_adaptive_binarize=args.adaptive_binarize,
        weighted_sampler_fn=create_weighted_sampler,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )
