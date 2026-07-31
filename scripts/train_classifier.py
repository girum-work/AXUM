"""
AXUM ROVER — Artefact Classifier Training
==========================================
Trains the MobileNetV2 artefact classification model.

Prerequisites:
  - 30+ images per class in data/artefact_classes/

Runtime: 1-3 hours on CPU

Run: python scripts/train_classifier.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.object_detection.detector import (
    train_artefact_classifier, ArtefactDataset
)
from config import OBJ_CLASSES

DATA_DIR  = "data/artefact_classes"
SAVE_PATH = Path("models/artefact_classifier.pth")

print("="*60)
print("ARTEFACT CLASSIFIER TRAINING (MobileNetV2)")
print("="*60)
print(f"Classes: {OBJ_CLASSES}\n")

# Check dataset
from pathlib import Path as P
total = 0
for cls in OBJ_CLASSES:
    cls_dir = P(DATA_DIR) / cls
    count   = len(list(cls_dir.glob("*.jpg")) +
                  list(cls_dir.glob("*.png"))) if cls_dir.exists() else 0
    status  = "✓" if count >= 30 else "✗"
    print(f"  {status} {cls:<25} {count} images")
    total  += count

print(f"\nTotal: {total} images")

if total < 50:
    print("\n✗ Too few images. Collect at least 30 per class first.")
    sys.exit(1)

SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

print("\nStarting training...\n")
train_artefact_classifier(
    data_dir      = DATA_DIR,
    save_path     = SAVE_PATH,
    num_epochs    = 40,
    batch_size    = 16,
    learning_rate = 0.001
)

print(f"\n✓ Model saved: {SAVE_PATH}")
print(f"\nNext: python scripts/train_yolo11.py")