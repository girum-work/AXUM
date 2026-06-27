"""
AXUM ROVER — Ge'ez OCR Package
================================
WHAT:  Recognize Ge'ez (Ethiopic) characters from inscription images.
WHY:   Core cultural AI feature — read Aksumite and church inscriptions on artefacts.

Exports:
    GeezOCRModel      — CNN + BiLSTM + CTC architecture (model.py)
    GeezOCRPipeline   — preprocess + predict wrapper (pipeline.py)
    train_ocr_model   — training loop (pipeline.py)
    load/save_ocr_model — weight I/O (model.py)

Train:  python scripts/train_ocr.py  →  models/geez_ocr.pth
Restore: src/ocr/llm_restoration.py fills [MISSING] tokens after OCR
"""

from src.ocr.model    import GeezOCRModel, load_ocr_model, save_ocr_model
from src.ocr.pipeline import GeezOCRPipeline, train_ocr_model

__all__ = [
    "GeezOCRModel",
    "GeezOCRPipeline",
    "load_ocr_model",
    "save_ocr_model",
    "train_ocr_model"
]
