"""
AXUM ROVER — YOLOv8 Fine-Tuning
=================================
Fine-tunes YOLOv8 nano on your artefact images.

Prerequisites:
  - pip install ultralytics
  - 30+ images per class in data/artefact_classes/

Runtime: 20-60 minutes on CPU

Run: python scripts/train_yolov8.py
"""

import sys, shutil, random, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OBJ_CLASSES

SOURCE_DIR  = Path("data/artefact_classes")
YOLO_DIR    = Path("data/yolo_dataset")
SAVE_PATH   = Path("models/yolov8_artefacts.pt")
TRAIN_SPLIT = 0.85
EPOCHS      = 50
IMG_SIZE    = 224

print("="*60)
print("YOLOV8 NANO FINE-TUNING")
print("="*60)

# Build YOLO dataset structure
for split in ["train", "val"]:
    (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

for class_idx, class_name in enumerate(OBJ_CLASSES):
    class_dir = SOURCE_DIR / class_name
    if not class_dir.exists():
        continue

    images    = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
    random.shuffle(images)
    split_idx = int(len(images) * TRAIN_SPLIT)
    splits    = {"train": images[:split_idx], "val": images[split_idx:]}

    for split_name, split_images in splits.items():
        for img_path in split_images:
            dest = YOLO_DIR / "images" / split_name / img_path.name
            shutil.copy(img_path, dest)
            label_path = (YOLO_DIR / "labels" / split_name /
                          img_path.with_suffix(".txt").name)
            with open(label_path, "w") as f:
                f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")

# Write dataset.yaml
dataset_yaml = {
    "path":  str(YOLO_DIR.absolute()),
    "train": "images/train",
    "val":   "images/val",
    "nc":    len(OBJ_CLASSES),
    "names": OBJ_CLASSES
}
yaml_path = YOLO_DIR / "dataset.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(dataset_yaml, f, default_flow_style=False)

print(f"Dataset prepared: {YOLO_DIR}")

# Train
try:
    from ultralytics import YOLO
    model   = YOLO("yolov8n.pt")
    results = model.train(
        data        = str(yaml_path),
        epochs      = EPOCHS,
        imgsz       = IMG_SIZE,
        batch       = 8,
        device      = "cpu",
        patience    = 10,
        project     = "models",
        name        = "yolov8_artefacts",
        exist_ok    = True,
        verbose     = True
    )
    best = Path("models/yolov8_artefacts/weights/best.pt")
    if best.exists():
        shutil.copy(best, SAVE_PATH)
        print(f"\n✓ YOLOv8 model saved: {SAVE_PATH}")
except ImportError:
    print("✗ ultralytics not installed. Run: pip install ultralytics")