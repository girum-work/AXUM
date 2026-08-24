"""Evaluate an AXUM OCR checkpoint on official HHD-Ethiopic test sets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr.model import IDX_TO_CHAR, load_ocr_model
from src.ocr.pipeline import HHDEthiopicDataset, ctc_collate_fn
from src.ocr.training_contract import compute_sequence_metrics


ROOT = Path(__file__).parent.parent
SPLITS = {
    "iid": (
        ROOT / "data/geez_characters/test/test_rand/image_text_pairs_test_rand.csv",
        ROOT / "data/geez_characters/test/test_rand/image_rand",
    ),
    "ood-18th": (
        ROOT / "data/geez_characters/test/test_18th/image_text_pairs_test_18th.csv",
        ROOT / "data/geez_characters/test/test_18th/image_18th",
    ),
}


def evaluate_split(model, device: torch.device, name: str, batch_size: int, beam: bool) -> dict:
    manifest, image_dir = SPLITS[name]
    dataset = HHDEthiopicDataset(
        str(manifest.parent),
        split="test",
        augment=False,
        manifest_path=manifest,
        image_dir=image_dir,
    )
    if not dataset:
        raise RuntimeError(f"No valid samples loaded for {name}: {manifest}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=ctc_collate_fn)

    references: list[str] = []
    predictions: list[str] = []
    model.eval()
    with torch.no_grad():
        for images, labels, _input_lengths, label_lengths in loader:
            log_probs = model(images.to(device))
            texts = model.decode_beam(log_probs) if beam else model.decode_greedy(log_probs)
            offset = 0
            for text, length_tensor in zip(texts, label_lengths):
                length = int(length_tensor.item())
                target = labels[offset : offset + length].tolist()
                offset += length
                references.append("".join(IDX_TO_CHAR[index] for index in target))
                predictions.append(text)

    metrics = compute_sequence_metrics(references, predictions).to_dict()
    metrics.update({
        "split": name,
        "manifest": str(manifest.relative_to(ROOT)),
        "loaded_samples": len(dataset),
        "decoder": "beam" if beam else "greedy",
    })
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate AXUM OCR on official HHD splits")
    parser.add_argument("--model", type=Path, default=ROOT / "models/geez_ocr.pth")
    parser.add_argument("--split", choices=["iid", "ood-18th", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--beam", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/ocr_official_evaluation.json")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    device = torch.device(args.device)
    model = load_ocr_model(args.model).to(device)
    selected = list(SPLITS) if args.split == "both" else [args.split]
    results = [evaluate_split(model, device, name, args.batch_size, args.beam) for name in selected]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for result in results:
        print(
            f"{result['split']}: CER={result['cer']:.2%} "
            f"CharAcc={result['character_accuracy']:.2%} "
            f"SeqAcc={result['sequence_accuracy']:.2%} "
            f"n={result['loaded_samples']}"
        )
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
