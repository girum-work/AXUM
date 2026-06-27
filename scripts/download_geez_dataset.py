# scripts/download_geez_dataset.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr.pipeline import download_hhd_ethiopic, verify_dataset

print("Downloading HHD-Ethiopic dataset...")
print("This will download ~300-500MB. Do not interrupt.\n")

download_hhd_ethiopic("data/geez_characters")

print("\nVerifying download...")
verify_dataset("data/geez_characters")