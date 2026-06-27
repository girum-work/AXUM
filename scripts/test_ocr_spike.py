# test_ocr_spike.py
import sys, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.ocr.pipeline import GeezOCRPipeline

pipeline = GeezOCRPipeline()
pipeline.load_model()

# Test on a simple Ge'ez character image
# Download a test image from Google: "Ge'ez inscription Ethiopia"
# Save it to: data/test_inscription.jpg

frame = cv2.imread("data/test_inscription.jpg")
if frame is not None:
    result = pipeline.process_image(frame)
    print(f"Regions detected: {result['total_regions']}")
    for r in result['regions']:
        print(f"  Text: '{r['text']}' | Confidence: {r['confidence']:.0%}")
        print(f"  Translation: {r['translation']['translation_en']}")
else:
    print("No test image found - save a Ge'ez inscription image to data/test_inscription.jpg")