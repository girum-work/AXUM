"""
AXUM ROVER - Dependency smoke test.

WHAT: Imports every critical third-party library and prints versions.
WHY: Confirms a fresh pip install succeeded before training or running src/.
WHEN: Run once after `pip install -r requirements.txt`.
"""

from __future__ import annotations

import importlib.metadata
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    """
    Run dependency imports and return a process exit code.

    Returns:
        0 when all critical imports succeed, otherwise 1.
    """
    print(f"Python version: {sys.version}")
    errors: list[str] = []

    try:
        import cv2
        print(f"OK OpenCV: {cv2.__version__}")
    except ImportError as exc:
        errors.append(f"FAIL OpenCV: {exc}")

    try:
        import numpy as np
        print(f"OK NumPy: {np.__version__}")
    except ImportError as exc:
        errors.append(f"FAIL NumPy: {exc}")

    try:
        import torch
        print(f"OK PyTorch: {torch.__version__}")
        print("  CPU available: True")
        print(f"  CUDA available: {torch.cuda.is_available()} (expected: False)")
    except ImportError as exc:
        errors.append(f"FAIL PyTorch: {exc}")

    try:
        import torchvision
        print(f"OK TorchVision: {torchvision.__version__}")
    except ImportError as exc:
        errors.append(f"FAIL TorchVision: {exc}")

    try:
        import matplotlib
        print(f"OK Matplotlib: {matplotlib.__version__}")
    except ImportError as exc:
        errors.append(f"FAIL Matplotlib: {exc}")

    try:
        import sklearn
        print(f"OK Scikit-learn: {sklearn.__version__}")
    except ImportError as exc:
        errors.append(f"FAIL Scikit-learn: {exc}")

    try:
        import librosa
        print(f"OK Librosa: {librosa.__version__}")
    except ImportError as exc:
        errors.append(f"FAIL Librosa: {exc}")

    try:
        import serial
        print(f"OK PySerial: {serial.__version__}")
    except ImportError as exc:
        errors.append(f"FAIL PySerial: {exc}")

    try:
        flask_version = importlib.metadata.version("flask")
        import flask  # noqa: F401
        print(f"OK Flask: {flask_version}")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        errors.append(f"FAIL Flask: {exc}")

    try:
        import reportlab
        print(f"OK ReportLab: {reportlab.Version}")
    except ImportError as exc:
        errors.append(f"FAIL ReportLab: {exc}")

    print("\n" + "=" * 50)
    if errors:
        print(f"FAILED ({len(errors)} errors):")
        for error in errors:
            print(f"  {error}")
        return 1

    print("ALL LIBRARIES INSTALLED SUCCESSFULLY")
    print("Ready to build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
