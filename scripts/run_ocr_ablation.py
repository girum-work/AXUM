# scripts/run_ocr_ablation.py
"""
Run incremental OCR training ablation: baseline → Fix1 → Fix2 → Fix3 → Fix4.

Each stage trains 5 epochs with cumulative fixes enabled.
Results are written to logs/ocr_ablation_results.json.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_ocr_errors import analyze_errors
from scripts.train_ocr import create_weighted_sampler
from src.ocr.pipeline import train_ocr_model

DATA_DIR = "data/geez_characters"
MODEL_PATH = Path("models/geez_ocr.pth")
RESULTS_PATH = Path("logs/ocr_ablation_results.json")
EPOCHS = 5
BATCH_SIZE = 32
LR = 0.001

# Cap batches per epoch for CPU ablation (~8 min/epoch at 150 train + 50 val batches)
MAX_TRAIN_BATCHES = 150
MAX_VAL_BATCHES = 50

STAGES = [
    {
        "name": "baseline",
        "label": "Baseline (no fixes)",
        "flags": {
            "use_weighted_sampler": False,
            "use_beam_val_decode": False,
            "use_stone_augment": False,
            "use_adaptive_binarize": False,
        },
    },
    {
        "name": "fix1_weighted_sampler",
        "label": "After Fix 1 — WeightedRandomSampler",
        "flags": {
            "use_weighted_sampler": True,
            "use_beam_val_decode": False,
            "use_stone_augment": False,
            "use_adaptive_binarize": False,
        },
    },
    {
        "name": "fix2_beam_decode",
        "label": "After Fix 2 — CTC beam search (val metric)",
        "flags": {
            "use_weighted_sampler": True,
            "use_beam_val_decode": True,
            "use_stone_augment": False,
            "use_adaptive_binarize": False,
        },
    },
    {
        "name": "fix3_stone_augment",
        "label": "After Fix 3 — Stone inscription augmentation",
        "flags": {
            "use_weighted_sampler": True,
            "use_beam_val_decode": True,
            "use_stone_augment": True,
            "use_adaptive_binarize": False,
        },
    },
    {
        "name": "fix4_adaptive_binarize",
        "label": "After Fix 4 — Adaptive binarization (inference preprocess)",
        "flags": {
            "use_weighted_sampler": True,
            "use_beam_val_decode": True,
            "use_stone_augment": True,
            "use_adaptive_binarize": True,
        },
    },
]


def run_ablation():
    """
    Execute all ablation stages and collect epoch + error-analysis metrics.

    Returns:
        dict with baseline error analysis and per-stage training results.
    """
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "timestamp": datetime.now().isoformat(),
        "epochs_per_stage": EPOCHS,
        "max_train_batches": MAX_TRAIN_BATCHES,
        "max_val_batches": MAX_VAL_BATCHES,
        "baseline_error_analysis": None,
        "stages": [],
    }

    print("=" * 70)
    print("OCR ABLATION — Step 0: Baseline error analysis")
    print("=" * 70)
    if MODEL_PATH.exists():
        results["baseline_error_analysis"] = analyze_errors(
            str(MODEL_PATH), DATA_DIR, max_batches=MAX_VAL_BATCHES
        )
    else:
        print("Baseline model not found — skipping error analysis.")

    for stage in STAGES:
        print("\n" + "=" * 70)
        print(f"TRAINING: {stage['label']}")
        print("=" * 70)

        stage_result = train_ocr_model(
            data_dir=DATA_DIR,
            save_path=MODEL_PATH,
            num_epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
            weighted_sampler_fn=create_weighted_sampler,
            max_train_batches=MAX_TRAIN_BATCHES,
            max_val_batches=MAX_VAL_BATCHES,
            **stage["flags"],
        )

        last_epoch = (
            stage_result["epoch_metrics"][-1]
            if stage_result and stage_result.get("epoch_metrics")
            else {}
        )

        inference_analysis = None
        if stage["flags"].get("use_adaptive_binarize") and MODEL_PATH.exists():
            inference_analysis = analyze_errors(
                str(MODEL_PATH), DATA_DIR, max_batches=MAX_VAL_BATCHES
            )

        results["stages"].append(
            {
                "name": stage["name"],
                "label": stage["label"],
                "flags": stage["flags"],
                "epoch_metrics": stage_result.get("epoch_metrics", []),
                "best_val_loss": stage_result.get("best_val_loss"),
                "final_epoch": last_epoch,
                "inference_error_analysis": inference_analysis,
            }
        )

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nStage complete — last epoch: {last_epoch}")
        print(f"Results saved to {RESULTS_PATH}")

    print_summary(results)
    return results


def print_summary(results: dict):
    """Print markdown-friendly summary table to stdout."""
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY TABLE")
    print("=" * 70)
    print(
        f"{'Stage':<28} {'TrainLoss':>10} {'ValLoss':>10} {'CharAcc':>10}"
    )
    print("-" * 70)

    if results.get("baseline_error_analysis"):
        ba = results["baseline_error_analysis"]["overall_char_accuracy"]
        print(f"{'Baseline (error analysis)':<28} {'—':>10} {'—':>10} {ba:>9.1%}")

    for stage in results.get("stages", []):
        fe = stage.get("final_epoch") or {}
        name = stage["name"][:28]
        print(
            f"{name:<28} "
            f"{fe.get('train_loss', 0):>10.4f} "
            f"{fe.get('val_loss', 0):>10.4f} "
            f"{fe.get('char_accuracy', 0):>9.1%}"
        )


if __name__ == "__main__":
    run_ablation()
