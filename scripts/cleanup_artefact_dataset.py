"""
AXUM ROVER — Audit and clean artefact classification images (all classes)
=========================================================================
Scans every folder under data/artefact_classes/, applies Ethiopian-focused
label rules, moves misclassified files, quarantines rejected media, and
updates metadata.csv.

Run after download or when you spot wrong labels (crosses in coin, paintings, etc.):

  python scripts/cleanup_artefact_dataset.py
  python scripts/cleanup_artefact_dataset.py --dry-run
  python scripts/cleanup_artefact_dataset.py --require-ethiopia
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ARTEFACT_CLASSES_DIR, ARTEFACT_METADATA_CSV, OBJ_CLASSES
from src.object_detection.artefact_label_rules import (
    LabelAction,
    decide_label,
    filename_to_title_hint,
)

QUARANTINE_DIR = ARTEFACT_CLASSES_DIR / "_rejected"
METADATA_FIELDS = [
    "filename", "class", "source_url", "license",
    "width_px", "height_px", "notes",
]


def _load_metadata() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not ARTEFACT_METADATA_CSV.exists():
        return rows
    with ARTEFACT_METADATA_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fn = row.get("filename", "").strip()
            if fn:
                rows[fn] = row
    return rows


def _save_metadata(rows: dict[str, dict]) -> None:
    sorted_rows = sorted(
        rows.values(),
        key=lambda r: (r.get("class", ""), r.get("filename", "")),
    )
    with ARTEFACT_METADATA_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        w.writeheader()
        for r in sorted_rows:
            w.writerow({k: r.get(k, "") for k in METADATA_FIELDS})


def _next_path(dest_dir: Path, filename: str) -> Path:
    """Avoid overwrite when moving into destination folder."""
    dest = dest_dir / filename
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    n = 1
    while True:
        candidate = dest_dir / f"{stem}_moved{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def audit_dataset(
    dry_run: bool = False,
    require_ethiopia: bool = True,
) -> dict[str, int]:
    """
    Walk all class folders; move/reject mislabelled images globally.

    Args:
        dry_run: Log actions without moving files
        require_ethiopia: Apply strict Ethiopian context filter

    Returns:
        Counts dict: kept, moved, rejected, per-class deltas
    """
    metadata = _load_metadata()
    stats: dict[str, int] = defaultdict(int)

    if not dry_run:
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

    for cls in OBJ_CLASSES:
        class_dir = ARTEFACT_CLASSES_DIR / cls
        if not class_dir.is_dir():
            continue

        for path in sorted(class_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            fn = path.name
            row = metadata.get(fn, {})
            text = " ".join([
                fn,
                filename_to_title_hint(fn),
                row.get("notes", ""),
                row.get("source_url", ""),
                row.get("class", cls),
            ])

            decision = decide_label(
                text, cls, require_ethiopia=require_ethiopia,
            )

            if decision.action == LabelAction.KEEP:
                stats["kept"] += 1
                continue

            if decision.action == LabelAction.REJECT:
                stats["rejected"] += 1
                dest = QUARANTINE_DIR / fn
                logger.info(f"REJECT [{cls}] {fn} — {decision.reason}")
                if not dry_run:
                    shutil.move(str(path), str(_next_path(QUARANTINE_DIR, fn)))
                    if fn in metadata:
                        metadata[fn]["class"] = "_rejected"
                        metadata[fn]["notes"] = (
                            f"{metadata[fn].get('notes', '')}; "
                            f"cleanup:{decision.reason}"
                        ).strip("; ")
                continue

            if decision.action == LabelAction.MOVE and decision.target_class:
                stats["moved"] += 1
                target = decision.target_class
                dest_dir = ARTEFACT_CLASSES_DIR / target
                dest_dir.mkdir(parents=True, exist_ok=True)
                new_name = fn
                # Rename if filename embeds wrong class (optional clarity)
                logger.info(
                    f"MOVE [{cls} → {target}] {fn} — {decision.reason}",
                )
                if not dry_run:
                    dest_path = _next_path(dest_dir, new_name)
                    shutil.move(str(path), str(dest_path))
                    new_fn = dest_path.name
                    if fn in metadata:
                        del metadata[fn]
                    metadata[new_fn] = {
                        "filename": new_fn,
                        "class": target,
                        "source_url": row.get("source_url", ""),
                        "license": row.get("license", ""),
                        "width_px": row.get("width_px", ""),
                        "height_px": row.get("height_px", ""),
                        "notes": (
                            f"cleanup moved from {cls}; {decision.reason}; "
                            f"{row.get('notes', '')}"
                        ).strip(),
                    }

    if not dry_run:
        _save_metadata(metadata)

    return dict(stats)


def print_counts() -> None:
    """Per-class file counts after cleanup."""
    print("\nPer-class counts:")
    for cls in OBJ_CLASSES:
        d = ARTEFACT_CLASSES_DIR / cls
        n = sum(
            1 for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ) if d.exists() else 0
        print(f"  {cls}: {n}")
    q = QUARANTINE_DIR
    if q.exists():
        rn = sum(1 for p in q.iterdir() if p.is_file())
        print(f"  _rejected: {rn}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean mislabelled artefact images (all classes)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report only; do not move files",
    )
    parser.add_argument(
        "--require-ethiopia",
        action="store_true",
        default=True,
        help="Reject/move items without Ethiopian context (default: on)",
    )
    parser.add_argument(
        "--allow-global",
        action="store_true",
        help="Only reject paintings/etc.; do not require Ethiopia keyword",
    )
    args = parser.parse_args()

    require_eth = args.require_ethiopia and not args.allow_global

    logger.info("AXUM artefact dataset cleanup (all classes)")
    if args.dry_run:
        logger.info("DRY RUN — no files will be moved")

    stats = audit_dataset(dry_run=args.dry_run, require_ethiopia=require_eth)
    print("\nSummary:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print_counts()

    if args.dry_run:
        logger.info("Re-run without --dry-run to apply changes")
    else:
        logger.info(
            "Review data/artefact_classes/_rejected/ then re-run "
            "download_artefact_images.py for short classes",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
