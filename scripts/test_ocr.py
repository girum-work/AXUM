"""
AXUM ROVER — OCR Model Test
=============================
Tests the trained Ge'ez OCR model on inscription images.

Prerequisites:
  - models/geez_ocr.pth must exist (run train_ocr.py first)
  - At least one Ge'ez inscription image

Run: python scripts/test_ocr.py
"""

import sys, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_PATH  = Path("models/geez_ocr.pth")
TEST_IMAGE  = Path("data/test_inscription.jpg")

if not MODEL_PATH.exists():
    print(f"✗ Model not found: {MODEL_PATH}")
    print("  Run: python scripts/train_ocr.py first")
    sys.exit(1)

from src.ocr.pipeline import GeezOCRPipeline
from src.ocr.llm_restoration import GeezRestorationEngine

print("="*60)
print("GE'EZ OCR + RESTORATION TEST")
print("="*60)

# Load OCR pipeline
print("Loading OCR model...")
pipeline = GeezOCRPipeline()
pipeline.load_model(MODEL_PATH)
print("✓ OCR model loaded\n")

# Load restoration engine
print("Loading restoration engine...")
engine = GeezRestorationEngine(mode="auto")
print(f"✓ Restoration engine ready | {engine.get_status()}\n")

# Test on image
if not TEST_IMAGE.exists():
    print(f"⚠ No test image at {TEST_IMAGE}")
    print("  Download a Ge'ez inscription image and save it there")
    print("  Searching Google for 'Ge'ez inscription Ethiopia stone' works")
    sys.exit(0)

frame = cv2.imread(str(TEST_IMAGE))
print(f"Testing on: {TEST_IMAGE.name}")
print(f"Image size: {frame.shape[1]}×{frame.shape[0]}px\n")

result = pipeline.process_image(frame)

print(f"Regions detected: {result['total_regions']}")
print(f"Reliable regions: {result['reliable_regions']}\n")

for i, region in enumerate(result["regions"]):
    print(f"Region {i+1}:")
    print(f"  Raw OCR:    '{region['text']}'")
    print(f"  Confidence: {region['confidence']:.0%}")

    if "[MISSING]" in region.get("text", ""):
        restoration = engine.restore(
            damaged_text = region["text"],
            period       = "Aksumite/Zagwe",
            location     = "Ethiopia"
        )
        print(f"  Restored:   '{restoration.restored_text}'")
        print(f"  English:    '{restoration.translation}'")
        print(f"  Conf:       {restoration.confidence:.0%} ({restoration.mode_used})")
    elif region.get("translation"):
        trans = region["translation"]
        print(f"  English:    '{trans.get('translation_en', '')}'")
    print()

print("✓ OCR pipeline test complete")