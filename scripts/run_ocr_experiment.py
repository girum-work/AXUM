"""Run and record one reproducible AXUM OCR training experiment."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate_ocr import evaluate_split
from scripts.train_ocr import create_weighted_sampler
from src.ocr.model import GEEZ_CHARSET, load_ocr_model
from src.ocr.pipeline import train_ocr_model
from src.ocr.training_contract import charset_fingerprint


ROOT = Path(__file__).parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a reproducible AXUM OCR experiment")
    parser.add_argument("--name", required=True)
    parser.add_argument("--data-dir", default="data/geez_characters")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weighted-sampler", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stone-augment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--beam-validation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=3)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1)
    parser.add_argument("--evaluate-beam", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.initial_model and args.encoder_checkpoint:
        parser.error("--initial-model and --encoder-checkpoint are mutually exclusive")

    model_path = ROOT / "models/ocr_experiments" / f"{args.name}.pth"
    report_dir = ROOT / "logs/ocr_experiments" / args.name
    report_path = report_dir / "experiment.json"
    if model_path.exists() or report_path.exists():
        parser.error(f"Experiment already exists: {args.name}")
    report_dir.mkdir(parents=True, exist_ok=False)

    configuration = vars(args).copy()
    configuration.update({
        "data_dir": str(args.data_dir),
        "initial_model": str(args.initial_model) if args.initial_model else None,
        "encoder_checkpoint": str(args.encoder_checkpoint) if args.encoder_checkpoint else None,
        "model_path": str(model_path.relative_to(ROOT)),
        "charset_size": len(GEEZ_CHARSET),
        "charset_fingerprint": charset_fingerprint(GEEZ_CHARSET),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "started_at": datetime.now(timezone.utc).isoformat(),
    })
    report_path.write_text(
        json.dumps({"status": "running", "configuration": configuration}, indent=2),
        encoding="utf-8",
    )

    try:
        training = train_ocr_model(
            data_dir=args.data_dir,
            save_path=model_path,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            use_weighted_sampler=args.weighted_sampler,
            use_beam_val_decode=args.beam_validation,
            use_stone_augment=args.stone_augment,
            weighted_sampler_fn=create_weighted_sampler,
            device=args.device,
            initial_model=args.initial_model,
            encoder_checkpoint=args.encoder_checkpoint,
            freeze_encoder_epochs=args.freeze_encoder_epochs,
            encoder_lr_scale=args.encoder_lr_scale,
            seed=args.seed,
        )
        model = load_ocr_model(model_path).to(torch.device(args.device))
        evaluation = [
            evaluate_split(model, torch.device(args.device), split, args.batch_size, args.evaluate_beam)
            for split in ("iid", "ood-18th")
        ]
        report = {
            "status": "complete",
            "configuration": configuration,
            "training": training,
            "official_evaluation": evaluation,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        report_path.write_text(json.dumps({
            "status": "failed",
            "configuration": configuration,
            "error": repr(exc),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        raise

    print(f"Model:  {model_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
