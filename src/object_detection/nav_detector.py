"""Independent COCO YOLO11 detector for obstacle checks during arm motion."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from ultralytics import YOLO

from config import NAV_YOLO_CONFIDENCE_MIN, NAV_YOLO_WEIGHTS


class NavObstacleDetector:
    """Load and run a YOLO11 model that is separate from artefact detection."""

    def __init__(
        self,
        weights: str = NAV_YOLO_WEIGHTS,
        confidence_min: float = NAV_YOLO_CONFIDENCE_MIN,
    ) -> None:
        self.weights = weights
        self.confidence_min = confidence_min
        self.model = YOLO(weights)
        logger.info(f"Navigation obstacle YOLO loaded: {weights}")

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Return COCO detections suitable for a motion-safety decision."""
        results = self.model(frame, conf=self.confidence_min, verbose=False)
        if not results:
            return []
        result = results[0]
        if result.boxes is None:
            return []
        detections: list[dict[str, Any]] = []
        for index in range(len(result.boxes)):
            class_id = int(result.boxes.cls[index].item())
            detections.append({
                "class_id": class_id,
                "class_name": str(result.names.get(class_id, "unknown")),
                "confidence": float(result.boxes.conf[index].item()),
                "xyxy": [float(value) for value in result.boxes.xyxy[index].tolist()],
            })
        return detections
