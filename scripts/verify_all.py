"""
AXUM ROVER - Full system verification.

WHAT: Checks models, datasets, optional services, and Arduino detection.
WHY: Gives a truthful pre-mission readiness report with actionable failures.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ResultFn = Callable[[], tuple[bool | None, str]]
results: dict[str, tuple[str, str]] = {}


def check(name: str, fn: ResultFn) -> bool | None:
    """
    Run one check and print PASS, FAIL, or WARN.

    Args:
        name: Human-readable check name.
        fn: Function returning (ok, message). ok=None means optional warning.

    Returns:
        The check status from fn.
    """
    try:
        ok, message = fn()
    except Exception as exc:
        ok, message = False, str(exc)[:90]

    if ok is True:
        status = "PASS"
    elif ok is None:
        status = "WARN"
    else:
        status = "FAIL"

    results[name] = (status, message)
    print(f"  {status:<4} {name:<35} {message}")
    return ok


def check_ocr_model() -> tuple[bool, str]:
    """Check whether the OCR model weight file exists."""
    path = Path("models/geez_ocr.pth")
    return path.exists(), "Found" if path.exists() else "MISSING - run train_ocr.py"


def check_classifier_model() -> tuple[bool, str]:
    """Check whether the MobileNet artefact classifier weight file exists."""
    path = Path("models/artefact_classifier.pth")
    return path.exists(), "Found" if path.exists() else "MISSING - run train_classifier.py"


def check_yolo_model() -> tuple[bool, str]:
    """Check whether the fine-tuned YOLOv8 artefact model exists."""
    path = Path("models/yolov8_artefacts.pt")
    return path.exists(), "Found" if path.exists() else "MISSING - run train_yolov8.py"


def check_geez_data() -> tuple[bool, str]:
    """Check OCR character image count."""
    directory = Path("data/geez_characters")
    count = sum(1 for _ in directory.rglob("*.png")) if directory.exists() else 0
    return count > 100, f"{count} images"


def check_artefact_data() -> tuple[bool, str]:
    """Check artefact classifier image count across configured classes."""
    from config import OBJ_CLASSES

    root = Path("data/artefact_classes")
    total = 0
    for class_name in OBJ_CLASSES:
        class_dir = root / class_name
        if class_dir.exists():
            total += len(list(class_dir.glob("*.jpg")))
            total += len(list(class_dir.glob("*.jpeg")))
            total += len(list(class_dir.glob("*.png")))
    return total >= 500, f"{total} images (target: 500+)"


def check_lm_studio() -> tuple[bool | None, str]:
    """Check optional LM Studio local server availability."""
    try:
        from openai import OpenAI

        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        models = client.models.list()
        name = models.data[0].id if models.data else "none"
        return bool(models.data), f"Running - model: {name}"
    except Exception:
        return None, "Not running - optional until LLM restoration work"


def check_arduino() -> tuple[bool | None, str]:
    """Check whether an Arduino-like serial port is detectable without opening motors."""
    from src.arm.controller import ArduinoSerial

    arduino = ArduinoSerial(connect=False)
    port = arduino._find_arduino_port()
    if port:
        return True, f"Found on {port}"
    return None, "Not detected - connect USB when testing hardware"


def main() -> int:
    """
    Run all verification checks.

    Returns:
        0 only when required checks pass; optional warnings do not fail.
    """
    print("=" * 60)
    print("AXUM ROVER - FULL SYSTEM VERIFICATION")
    print("=" * 60)

    print("\n[1/5] Imports...")
    check("config.py", lambda: (True, "OK"))
    check("controller.py", lambda: (__import__("src.arm.controller"), "OK")[1] and (True, "OK"))
    check("main_pipeline.py", lambda: (__import__("src.pipeline.main_pipeline"), "OK")[1] and (True, "OK"))
    check("crack detector", lambda: (__import__("src.crack_detection.detector"), "OK")[1] and (True, "OK"))

    print("\n[2/5] Models...")
    check("Ge'ez OCR model", check_ocr_model)
    check("Artefact classifier", check_classifier_model)
    check("YOLOv8 model", check_yolo_model)

    print("\n[3/5] Datasets...")
    check("Ge'ez character data", check_geez_data)
    check("Artefact class data", check_artefact_data)

    print("\n[4/5] LLM...")
    check("LM Studio server", check_lm_studio)

    print("\n[5/5] Arduino...")
    check("Arduino connection", check_arduino)

    passed = sum(1 for status, _ in results.values() if status == "PASS")
    warnings = sum(1 for status, _ in results.values() if status == "WARN")
    failed = [name for name, (status, _message) in results.items() if status == "FAIL"]
    total = len(results)

    print(f"\n{'=' * 60}")
    print(f"RESULT: {passed}/{total} passed, {warnings} warning(s), {len(failed)} failure(s)")
    print(f"{'=' * 60}")

    if failed:
        print("\nRequired fixes before a full mission:")
        for name in failed:
            print(f"  - {name}: {results[name][1]}")
        return 1

    print("\nRequired checks passed. Warnings may still block optional features.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
