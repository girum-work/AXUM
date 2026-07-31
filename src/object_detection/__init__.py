"""
AXUM ROVER — Object Detection & Classification
==============================================
WHAT:  Find artefacts in the tray (OpenCV) and classify type (MobileNetV2 / YOLO).
WHY:   Arm needs pixel coordinates (contours) AND category (pottery, coin, etc.).

Two-stage pipeline:
    detect_objects_in_tray()     — Stage 1: bounding boxes, no ML
    detect_and_classify_tray()   — Stage 1 + Stage 2: full pipeline

Train classifier: python scripts/train_classifier.py
Train YOLO:       python scripts/train_yolo11.py
Coins:            src/object_detection/coin_inspection.py (second stage)
"""

from src.object_detection.detector import (
    ArtefactClassifier,
    YOLOArtefactClassifier,
    ArtefactDatabase,
    detect_objects_in_tray,
    detect_and_classify_tray,
    draw_detection_overlay
)

__all__ = [
    "ArtefactClassifier",
    "YOLOArtefactClassifier",
    "ArtefactDatabase",
    "detect_objects_in_tray",
    "detect_and_classify_tray",
    "draw_detection_overlay"
]
