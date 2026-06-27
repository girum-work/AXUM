"""
AXUM ROVER — Crack Detection Spike Test
=========================================
Tests the crack detection pipeline on downloaded stone images.
Run this BEFORE any model training to verify the pipeline works.

Usage:
  1. Download 20-30 stone surface images to data/crack_images/test/
     (cracked and intact — mix of both)
  2. Run: python scripts/test_crack_spike.py

A results window will open showing detected cracks.
Adjust config.py thresholds if results look wrong.
"""

import sys, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

TEST_DIR = Path("data/crack_images/test")

if not TEST_DIR.exists() or not list(TEST_DIR.glob("*.jpg")):
    print("✗ No test images found.")
    print(f"  Create folder: {TEST_DIR}")
    print("  Add 20-30 stone surface images (JPG/PNG)")
    print("  Mix of cracked and intact surfaces")
    sys.exit(1)

print("="*60)
print("CRACK DETECTION SPIKE TEST")
print("="*60)

# Import after path setup
from src.crack_detection.detector import CrackDetector

detector = CrackDetector()
images   = list(TEST_DIR.glob("*.jpg")) + list(TEST_DIR.glob("*.png"))
print(f"Testing on {len(images)} images...\n")

print(f"{'Image':<40} {'Cracks':>6} {'Severity':>9} {'Assessment'}")
print("─" * 70)

total_cracks    = 0
false_positives = 0

for img_path in sorted(images):
    frame  = cv2.imread(str(img_path))
    if frame is None:
        continue

    result = detector.detect(frame)
    cracks = result.get("crack_count", 0)
    sev    = result.get("severity_score", 0.0)

    if "intact" in img_path.name.lower() and cracks > 5:
        assessment      = "⚠ FALSE POSITIVE"
        false_positives += 1
    elif cracks == 0 and "crack" in img_path.name.lower():
        assessment = "⚠ MISSED CRACK"
    elif sev > 0.5:
        assessment = "Critical damage"
    elif sev > 0.2:
        assessment = "Moderate damage"
    elif cracks > 0:
        assessment = "Minor cracks"
    else:
        assessment = "Intact"

    total_cracks += cracks
    print(f"{img_path.name:<40} {cracks:>6} {sev:>9.3f} {assessment}")

print("─" * 70)
print(f"\nTotal cracks detected: {total_cracks}")
print(f"Potential false positives: {false_positives}")

if false_positives > len(images) * 0.3:
    print("\n⚠ Too many false positives.")
    print("  Fix: Open config.py, increase MIN_ASPECT_RATIO to 4.5 or 5.0")
else:
    print("\n✓ Crack detection working. Proceed to OCR training.")