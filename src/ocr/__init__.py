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

# Imported lazily (PEP 562). Eager imports pulled pipeline -> object_detection
# -> ultralytics, so `from src.ocr.damage import ...` needed the whole vision
# stack despite damage.py being pure stdlib.
_LAZY = {
    "GeezOCRModel": "src.ocr.model",
    "load_ocr_model": "src.ocr.model",
    "save_ocr_model": "src.ocr.model",
    "GeezOCRPipeline": "src.ocr.pipeline",
    "train_ocr_model": "src.ocr.pipeline",
}


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "GeezOCRModel",
    "GeezOCRPipeline",
    "load_ocr_model",
    "save_ocr_model",
    "train_ocr_model"
]
