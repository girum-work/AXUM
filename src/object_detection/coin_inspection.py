"""
AXUM ROVER — Ethiopian coin inspection (second-stage YOLO + wear heuristic)
==========================================================================
After the tray classifier detects ``coin``, this module identifies:

  - Era: Aksumite, Menelik, Haile Selassie, modern birr/cent, etc.
  - Whether to run Ge'ez OCR (heavily worn coins → ancient inscription candidate)

Design rationale (competition narrative):
    Modern circulation coins are sharp and legible; museum Aksumite coins are
    often worn. High wear + low OCR legibility maps toward ``coin_aksumite`` or
    ``coin_wear_unknown_ancient`` and triggers inscription scanning — matching
    how a conservator treats faded specimens differently from shiny birr.

Author: Axum Rover Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    COIN_DAMAGE_ANCIENT_THRESHOLD,
    COIN_OCR_SCAN_CONFIDENCE_MAX,
    OBJ_CONFIDENCE_MIN,
    YOLO_COIN_CLASSES,
    YOLO_COIN_MODEL_PATH,
)

# YOLO classes that imply Ge'ez / ancient epigraphy follow-up
ANCIENT_COIN_CLASSES = frozenset({
    "coin_aksumite",
    "coin_wear_unknown_ancient",
})


@dataclass
class CoinInspectionResult:
    """
    Full second-stage result for one coin crop.

    Attributes:
        era_class: YOLO coin subtype or heuristic label
        era_confidence: Detector confidence 0–1
        denomination_hint: Parsed hint e.g. '1 birr', '25 cent', or ''
        wear_score: 0–1 (higher = more worn / faded)
        recommend_geez_ocr: True when inscription scan should run
        reasoning: Short explanation for dashboard / catalogue
        source: 'yolo_coin' | 'heuristic_wear' | 'unknown'
    """

    era_class: str = "coin_unknown"
    era_confidence: float = 0.0
    denomination_hint: str = ""
    wear_score: float = 0.0
    recommend_geez_ocr: bool = False
    reasoning: str = ""
    source: str = "unknown"
    yolo_available: bool = False


def compute_wear_score(crop_bgr: np.ndarray) -> float:
    """
    Estimate coin surface wear from blur and low local contrast.

    Higher scores suggest ancient/worn specimens (heuristic only).

    Args:
        crop_bgr: BGR image of coin on turntable

    Returns:
        Float in [0, 1]
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    # Laplacian variance — low = blurry/worn
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_component = 1.0 - min(lap_var / 200.0, 1.0)
    # Low global contrast — faded relief
    contrast = float(gray.std()) / 128.0
    fade_component = 1.0 - min(contrast, 1.0)
    return round(0.6 * blur_component + 0.4 * fade_component, 3)


def denomination_from_class(era_class: str) -> str:
    """
    Map YOLO subtype name to human-readable denomination hint.

    Args:
        era_class: e.g. coin_modern_birr

    Returns:
        English hint string
    """
    mapping = {
        "coin_modern_birr": "Ethiopian birr (various)",
        "coin_modern_cent": "Ethiopian cent/santim",
        "coin_menelik": "Menelik II era",
        "coin_haile_selassie": "Haile Selassie era",
        "coin_aksumite": "Aksumite / ancient",
        "coin_wear_unknown_ancient": "Ancient (wear — uncertain era)",
        "coin_modern_other": "Modern Ethiopian",
    }
    return mapping.get(era_class, "")


class CoinInspectionPipeline:
    """
    Second-stage coin analysis using optional YOLO + wear heuristics.

    Usage:
        pipe = CoinInspectionPipeline()
        result = pipe.inspect(coin_crop_bgr, ocr_confidence=0.3)
    """

    def __init__(self, model_path: Path | None = None):
        """
        Args:
            model_path: Path to yolo11_coins.pt (None = config default)
        """
        self.model_path = Path(model_path or YOLO_COIN_MODEL_PATH)
        self.model = None
        self.model_loaded = False
        self._load_model()

    def _load_model(self) -> None:
        """Load coin-detail YOLO if weights exist."""
        if not self.model_path.exists():
            logger.info(
                f"Coin YOLO not found at {self.model_path} — "
                "using wear heuristics only. Train with train_yolo11_coins.py"
            )
            return
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))
            self.model_loaded = True
            logger.info(f"Coin detail YOLO loaded: {self.model_path}")
        except ImportError:
            logger.warning("ultralytics missing — coin YOLO unavailable")
        except Exception as exc:
            logger.warning(f"Coin YOLO load failed: {exc}")

    def inspect(
        self,
        crop_bgr: np.ndarray,
        ocr_confidence: float | None = None,
    ) -> CoinInspectionResult:
        """
        Classify coin era and decide whether to run Ge'ez OCR.

        Logic:
            1. YOLO coin subtype if model loaded
            2. wear_score from image heuristics
            3. If wear high OR YOLO says ancient OR OCR conf low → recommend OCR

        Args:
            crop_bgr: Cropped coin image from tray
            ocr_confidence: Optional pre-run OCR confidence on crop (0–1)

        Returns:
            CoinInspectionResult
        """
        wear = compute_wear_score(crop_bgr)
        era_class = "coin_unknown"
        era_conf = 0.0
        source = "heuristic_wear"

        if self.model_loaded and self.model is not None:
            try:
                results = self.model(crop_bgr, verbose=False, conf=0.20)
                if results and len(results) > 0:
                    r0 = results[0]
                    boxes = r0.boxes
                    if boxes is not None and len(boxes) > 0:
                        confs = boxes.conf.cpu().numpy()
                        idx = int(confs.argmax())
                        era_conf = float(confs[idx])
                        ci = int(boxes.cls[idx].item())
                        era_class = r0.names.get(ci, "coin_unknown")
                        source = "yolo_coin"
            except Exception as exc:
                logger.debug(f"Coin YOLO inference failed: {exc}")

        # Wear override: very worn → ancient candidate if YOLO uncertain
        if wear >= COIN_DAMAGE_ANCIENT_THRESHOLD and era_conf < OBJ_CONFIDENCE_MIN:
            era_class = "coin_wear_unknown_ancient"
            era_conf = max(era_conf, wear)
            source = "heuristic_wear"
            reasoning = (
                f"High wear score ({wear:.2f}) — treating as ancient specimen "
                "for Ge'ez inscription scan."
            )
        elif era_class in ANCIENT_COIN_CLASSES:
            reasoning = (
                f"Classified as {era_class} ({era_conf:.0%}). "
                "Ancient coin — recommend epigraphic scan."
            )
        else:
            reasoning = (
                f"Classified as {era_class} ({era_conf:.0%}). "
                f"Wear {wear:.2f}."
            )

        recommend_ocr = (
            era_class in ANCIENT_COIN_CLASSES
            or wear >= COIN_DAMAGE_ANCIENT_THRESHOLD
            or (
                ocr_confidence is not None
                and ocr_confidence < COIN_OCR_SCAN_CONFIDENCE_MAX
                and wear > 0.35
            )
        )

        if recommend_ocr and "recommend" not in reasoning.lower():
            reasoning += " Ge'ez OCR recommended."

        return CoinInspectionResult(
            era_class=era_class,
            era_confidence=era_conf,
            denomination_hint=denomination_from_class(era_class),
            wear_score=wear,
            recommend_geez_ocr=recommend_ocr,
            reasoning=reasoning.strip(),
            source=source,
            yolo_available=self.model_loaded,
        )
