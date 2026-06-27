# scripts/analyze_ocr_errors.py
"""
Run after initial training to identify which characters
your model gets wrong most often.
This tells you WHERE to focus your improvement efforts.
"""
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr.model import GeezOCRModel, IDX_TO_CHAR, load_ocr_model  # noqa: F401
from src.ocr.pipeline import HHDEthiopicDataset, ctc_collate_fn


def analyze_errors(model_path: str, data_dir: str, max_batches: int = None):
    """
    Character-level error analysis on the validation split.

    WHAT: Compares greedy-decoded predictions to ground-truth labels per
    Ge'ez character and reports worst/best performers.
    WHY: Pinpoints class imbalance vs. augmentation vs. decoding issues
    before applying training fixes.

    Args:
        model_path: Path to .pth checkpoint
        data_dir: Path to data/geez_characters/
        max_batches: Optional cap on val batches (None = full val set)
    """
    model = load_ocr_model(model_path)
    model.eval()

    val_dataset = HHDEthiopicDataset(data_dir, split="val", augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=ctc_collate_fn,
    )

    char_correct = defaultdict(int)
    char_total = defaultdict(int)

    with torch.no_grad():
        for batch_idx, (images, labels, input_lengths, label_lengths) in enumerate(
            val_loader
        ):
            if max_batches is not None and batch_idx >= max_batches:
                break
            log_probs = model(images)
            texts = model.decode_greedy(log_probs)

            label_offset = 0
            for b, pred_text in enumerate(texts):
                ll = label_lengths[b].item()
                true_ids = labels[label_offset : label_offset + ll].tolist()
                true_text = "".join(IDX_TO_CHAR.get(i, "") for i in true_ids)
                label_offset += ll

                for tc in true_text:
                    char_total[tc] += 1
                    if tc in pred_text:
                        char_correct[tc] += 1

    char_accuracy = {
        c: char_correct[c] / char_total[c]
        for c in char_total
        if char_total[c] >= 5
    }

    sorted_chars = sorted(char_accuracy.items(), key=lambda x: x[1])

    print("\n" + "=" * 60)
    print("CHARACTER-LEVEL ERROR ANALYSIS")
    print("=" * 60)
    print(f"\nWorst 20 characters (lowest accuracy):")
    print(f"{'Char':<8} {'Accuracy':>10} {'Samples':>10}")
    print("-" * 30)
    for char, acc in sorted_chars[:20]:
        print(f"  {char:<6} {acc:>10.1%} {char_total[char]:>10}")

    print(f"\nBest 10 characters:")
    for char, acc in sorted_chars[-10:]:
        print(f"  {char:<6} {acc:>10.1%} {char_total[char]:>10}")

    total_correct = sum(char_correct.values())
    total_chars = sum(char_total.values())
    overall = total_correct / total_chars if total_chars else 0.0
    print(f"\nOverall character accuracy: {overall:.1%}")
    print(
        f"Characters with <50% accuracy: "
        f"{sum(1 for a in char_accuracy.values() if a < 0.5)}"
    )
    print(
        f"Characters with >90% accuracy: "
        f"{sum(1 for a in char_accuracy.values() if a > 0.9)}"
    )

    return {
        "overall_char_accuracy": overall,
        "char_accuracy": char_accuracy,
        "char_total": dict(char_total),
    }


if __name__ == "__main__":
    model_path = Path("models/geez_ocr.pth")
    if not model_path.exists():
        print(f"Baseline unavailable: {model_path} not found.")
        print("Train a model first with scripts/train_ocr.py")
        sys.exit(0)

    analyze_errors(str(model_path), "data/geez_characters")
