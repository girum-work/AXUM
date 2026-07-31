"""
AXUM ROVER — YOLO11 Coin Detail Training
=========================================
Second-stage detector: Ethiopian coin era and denomination.

Prerequisites:
  - Images in data/coin_subtypes/{coin_aksumite, coin_menelik, ...}/
  - Run: python scripts/download_artefact_images.py --coin-subtypes-only

Classes (config.YOLO_COIN_CLASSES):
  coin_aksumite, coin_menelik, coin_haile_selassie,
  coin_modern_birr, coin_modern_cent, coin_modern_other,
  coin_wear_unknown_ancient (optional manual folder for worn specimens)

Run: python scripts/train_yolo11_coins.py
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    COIN_SUBTYPES_DIR,
    YOLO_COIN_CLASSES,
    YOLO_COIN_DATASET_DIR,
    YOLO_COIN_MODEL_PATH,
)

SOURCE_DIR = COIN_SUBTYPES_DIR
TRAIN_SPLIT = 0.85
EPOCHS = 40
IMG_SIZE = 224


def prepare_yolo_coin_dataset() -> Path:
    """
    Copy coin_subtypes folders into YOLO train/val layout with bbox labels.

    Returns:
        Path to dataset.yaml
    """
    for split in ("train", "val"):
        (YOLO_COIN_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_COIN_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    present_classes: list[str] = []
    for class_name in YOLO_COIN_CLASSES:
        class_dir = SOURCE_DIR / class_name
        if not class_dir.is_dir():
            continue
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        if len(images) < 5:
            print(f"  ⚠ {class_name}: only {len(images)} images — add more or skip")
            continue
        present_classes.append(class_name)

    for class_idx, class_name in enumerate(present_classes):
        class_dir = SOURCE_DIR / class_name
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_SPLIT)
        splits = {"train": images[:split_idx], "val": images[split_idx:]}

        for split_name, split_images in splits.items():
            for img_path in split_images:
                dest = YOLO_COIN_DATASET_DIR / "images" / split_name / img_path.name
                shutil.copy2(img_path, dest)
                label_path = (
                    YOLO_COIN_DATASET_DIR / "labels" / split_name
                    / img_path.with_suffix(".txt").name
                )
                with open(label_path, "w", encoding="utf-8") as f:
                    # NOTE (Systems Integration Engineer): `yolo_idx` is not
                    # defined anywhere in this function -- pre-existing bug,
                    # unrelated to the YOLOv8->YOLO11 naming pass, found
                    # while doing that pass. NOT fixed here per instruction
                    # to flag rather than silently patch. Almost certainly
                    # should be `class_idx` (the loop variable just above),
                    # but that's a real behavioral call for whoever owns
                    # this script, not mine to guess and fix silently.
                    f.write(f"{yolo_idx} 0.5 0.5 1.0 1.0\n")

    if not present_classes:
        raise FileNotFoundError(
            f"No coin subtype images under {SOURCE_DIR}. "
            "Run: python scripts/download_artefact_images.py --coin-subtypes-only"
        )

    yaml_path = YOLO_COIN_DATASET_DIR / "dataset.yaml"
    data = {
        "path": str(YOLO_COIN_DATASET_DIR.absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(present_classes),
        "names": present_classes,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"Prepared {len(present_classes)} coin classes: {present_classes}")
    return yaml_path


def main() -> None:
    """Train YOLO11 nano on Ethiopian coin subtypes."""
    print("=" * 60)
    print("YOLO11 — ETHIOPIAN COIN DETAIL")
    print("=" * 60)

    yaml_path = prepare_yolo_coin_dataset()

    try:
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        model.train(
            data=str(yaml_path),
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=8,
            device="cpu",
            patience=10,
            project="models",
            name="yolo11_coins",
            exist_ok=True,
            verbose=True,
        )
        best = Path("models/yolo11_coins/weights/best.pt")
        if best.exists():
            shutil.copy(best, YOLO_COIN_MODEL_PATH)
            print(f"\n✓ Saved: {YOLO_COIN_MODEL_PATH}")
    except ImportError:
        print("✗ pip install ultralytics")


if __name__ == "__main__":
    main()
