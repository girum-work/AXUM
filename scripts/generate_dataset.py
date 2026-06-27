"""
AXUM ROVER — Ge'ez Restoration Dataset Generator
==================================================
Builds chat-format fine-tuning JSONL for Qwen2.5 / Unsloth QLoRA.

Sources:
  - Curated epigraphic corpus (src/ocr/corpus.py)
  - Optional HHD-Ethiopic CSV labels (--from-hhd)
  - User extensions in data/corpus/geez_inscriptions.jsonl

Outputs:
  - data/geez_restoration_train.jsonl
  - data/geez_restoration_val.jsonl
  - data/geez_restoration_eval.jsonl
  - data/geez_restoration_dataset.stats.json

Run:
  python scripts/generate_dataset.py
  python scripts/generate_dataset.py --from-hhd --examples-per-phrase 12 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    GEEZ_RESTORATION_CORPUS,
    GEEZ_RESTORATION_EVAL,
    GEEZ_RESTORATION_LEGACY,
    GEEZ_RESTORATION_TRAIN,
    GEEZ_RESTORATION_VAL,
    RESTORATION_DAMAGE_RATES,
    RESTORATION_EXAMPLES_PER_PHRASE,
    RESTORATION_SPLIT_SEED,
    RESTORATION_TRAIN_RATIO,
    RESTORATION_VAL_RATIO,
)
from src.ocr.corpus import (
    InscriptionEntry,
    build_full_corpus,
    save_corpus_json,
    split_phrase_ids,
)
from src.ocr.damage import (
    apply_damage,
    confidence_from_damage,
    count_content_graphemes,
    needs_expert_from_damage,
)
from src.ocr.restoration_prompts import build_chat_messages


def _build_reasoning(entry: InscriptionEntry, damage_mode: str) -> str:
    """Deterministic reasoning string from entry metadata."""
    return (
        f"Based on {entry.context or 'inscription context'}. "
        f"{entry.period} material from {entry.location} "
        f"commonly uses this formula (damage mode: {damage_mode})."
    )


def build_example(
    entry: InscriptionEntry,
    damage_rate: float,
    rng: random.Random,
) -> dict | None:
    """
    Build one training row with chat messages + metadata.

    Args:
        entry: Clean inscription record
        damage_rate: Erosion severity target
        rng: Random generator instance

    Returns:
        Row dict or None if damage invalid / duplicate skip
    """
    result = apply_damage(entry.text, damage_rate)
    damaged = result.damaged

    if damaged == entry.text or "[MISSING]" not in damaged:
        return None

    total_g = count_content_graphemes(entry.text)
    conf = confidence_from_damage(result.missing_count, total_g)
    expert = needs_expert_from_damage(result.missing_count, total_g)
    reasoning = _build_reasoning(entry, result.mode.value)

    messages = build_chat_messages(
        damaged_text=damaged,
        restored_text=entry.text,
        translation=entry.translation,
        period=entry.period,
        location=entry.location,
        confidence=conf,
        reasoning=reasoning,
        needs_expert=expert,
        include_few_shot=False,
    )

    return {
        "messages": messages,
        "metadata": {
            "phrase_id": entry.id,
            "source": entry.source,
            "clean_text": entry.text,
            "damaged_text": damaged,
            "damage_rate": damage_rate,
            "damage_mode": result.mode.value,
            "missing_count": result.missing_count,
        },
    }


def generate_splits(
    entries: list[InscriptionEntry],
    examples_per_phrase: int,
    damage_rates: tuple[float, ...],
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], int]:
    """
    Generate train/val/eval example lists without phrase leakage.

    Returns:
        (train_rows, val_rows, eval_rows, skipped_count)
    """
    rng = random.Random(seed)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    eval_rows: list[dict] = []
    seen_damaged: set[tuple[str, str, str]] = set()
    skipped = 0

    for entry in entries:
        if entry.id in test_ids:
            bucket_eval = eval_rows
            bucket_val = None
            bucket_train = None
        elif entry.id in val_ids:
            bucket_eval = None
            bucket_val = val_rows
            bucket_train = None
        elif entry.id in train_ids:
            bucket_eval = None
            bucket_val = None
            bucket_train = train_rows
        else:
            continue

        n_added = 0
        attempts = 0
        max_attempts = examples_per_phrase * 4

        while n_added < examples_per_phrase and attempts < max_attempts:
            attempts += 1
            rate = rng.choice(damage_rates)
            row = build_example(entry, rate, rng)
            if row is None:
                skipped += 1
                continue

            meta = row["metadata"]
            key = (meta["phrase_id"], meta["damaged_text"], meta["damage_mode"])
            if key in seen_damaged:
                skipped += 1
                continue
            seen_damaged.add(key)

            if bucket_train is not None:
                train_rows.append(row)
            elif bucket_val is not None:
                val_rows.append(row)
            elif bucket_eval is not None:
                eval_rows.append(row)
            n_added += 1

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(eval_rows)
    return train_rows, val_rows, eval_rows, skipped


def write_jsonl(rows: list[dict], path: Path, include_metadata: bool = False) -> None:
    """
    Write JSONL file for fine-tuning.

    Args:
        rows: Example dicts with 'messages' key
        path: Output file path
        include_metadata: If True, add metadata field per line (for debugging)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            out = {"messages": row["messages"]}
            if include_metadata:
                out["metadata"] = row.get("metadata", {})
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def generate_dataset(
    include_hhd: bool = True,
    examples_per_phrase: int = RESTORATION_EXAMPLES_PER_PHRASE,
    seed: int = RESTORATION_SPLIT_SEED,
    export_corpus: bool = True,
) -> dict:
    """
    Main dataset generation entry point.

    Args:
        include_hhd: Pull unique lines from HHD CSV
        examples_per_phrase: Synthetic variants per phrase ID
        seed: RNG seed for splits and damage
        export_corpus: Write merged corpus JSONL for editing

    Returns:
        Statistics dict
    """
    entries = build_full_corpus(include_hhd=include_hhd, include_json=True)
    if not entries:
        logger.error("No corpus entries — cannot generate dataset")
        return {"error": "empty corpus"}

    if export_corpus:
        curated_only = [e for e in entries if e.source == "curated"]
        save_corpus_json(entries, GEEZ_RESTORATION_CORPUS)
        logger.info(f"Corpus export: {len(entries)} lines ({len(curated_only)} curated)")

    train_ids, val_ids, test_ids = split_phrase_ids(
        entries,
        train_ratio=RESTORATION_TRAIN_RATIO,
        val_ratio=RESTORATION_VAL_RATIO,
        seed=seed,
    )

    train_rows, val_rows, eval_rows, skipped = generate_splits(
        entries,
        examples_per_phrase,
        RESTORATION_DAMAGE_RATES,
        train_ids,
        val_ids,
        test_ids,
        seed,
    )

    write_jsonl(train_rows, GEEZ_RESTORATION_TRAIN)
    write_jsonl(val_rows, GEEZ_RESTORATION_VAL)
    write_jsonl(eval_rows, GEEZ_RESTORATION_EVAL, include_metadata=True)

    # Legacy combined file (train only) for older Colab notebooks
    write_jsonl(train_rows, GEEZ_RESTORATION_LEGACY)

    stats = {
        "generated": datetime.now().isoformat(),
        "corpus_phrases": len(entries),
        "curated": sum(1 for e in entries if e.source == "curated"),
        "hhd": sum(1 for e in entries if e.source == "hhd"),
        "user": sum(1 for e in entries if e.source == "user"),
        "train_phrases": len(train_ids),
        "val_phrases": len(val_ids),
        "test_phrases": len(test_ids),
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "eval_examples": len(eval_rows),
        "examples_per_phrase": examples_per_phrase,
        "skipped": skipped,
        "damage_rates": list(RESTORATION_DAMAGE_RATES),
        "seed": seed,
        "paths": {
            "train": str(GEEZ_RESTORATION_TRAIN.resolve()),
            "val": str(GEEZ_RESTORATION_VAL.resolve()),
            "eval": str(GEEZ_RESTORATION_EVAL.resolve()),
            "corpus": str(GEEZ_RESTORATION_CORPUS.resolve()),
        },
    }

    stats_path = GEEZ_RESTORATION_TRAIN.with_name("geez_restoration_dataset.stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Ge'ez restoration fine-tuning data")
    parser.add_argument(
        "--from-hhd",
        action="store_true",
        default=True,
        help="Include HHD-Ethiopic CSV labels (default: on)",
    )
    parser.add_argument(
        "--no-hhd",
        action="store_true",
        help="Skip HHD CSV import (curated + user JSONL only)",
    )
    parser.add_argument(
        "--examples-per-phrase",
        type=int,
        default=RESTORATION_EXAMPLES_PER_PHRASE,
    )
    parser.add_argument("--seed", type=int, default=RESTORATION_SPLIT_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    include_hhd = args.from_hhd and not args.no_hhd

    logger.info("=" * 60)
    logger.info("GE'EZ RESTORATION DATASET GENERATOR")
    logger.info("=" * 60)

    stats = generate_dataset(
        include_hhd=include_hhd,
        examples_per_phrase=args.examples_per_phrase,
        seed=args.seed,
    )

    if "error" in stats:
        sys.exit(1)

    logger.info(f"Corpus phrases: {stats['corpus_phrases']} "
                f"(curated={stats['curated']}, hhd={stats['hhd']})")
    logger.info(f"Train examples: {stats['train_examples']}")
    logger.info(f"Val examples:   {stats['val_examples']}")
    logger.info(f"Eval examples:  {stats['eval_examples']} (held-out phrases)")
    logger.info(f"Saved train → {stats['paths']['train']}")

    # Sample
    with open(GEEZ_RESTORATION_TRAIN, encoding="utf-8") as f:
        line = f.readline()
        if line:
            sample = json.loads(line)
            user_msg = sample["messages"][1]["content"]
            asst = json.loads(sample["messages"][2]["content"])
            logger.info(f"Sample damaged: {user_msg.split('Damaged text:')[-1][:80]}...")
            logger.info(f"Sample restored: {asst['restored_text']}")

    logger.info("Next: python scripts/export_restoration_colab.py")
    logger.info("Then open notebooks/colab_geez_restoration_qlora.ipynb in Google Colab.")
