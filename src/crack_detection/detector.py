"""
AXUM ROVER - Crack detection module.

WHAT: Pure OpenCV crack extraction for stone, pottery, and inscription images.
WHY: Crack detection must work locally on CPU and before any ML model is
trained, so this module uses explainable image processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    BLUR_KERNEL,
    CANNY_T1,
    CANNY_T2,
    CRACK_SEVERITY_THRESHOLD,
    MIN_ASPECT_RATIO,
    MIN_CRACK_AREA,
    MIN_CRACK_LEN,
    USE_AUTO_CANNY,
)


@dataclass
class CrackResult:
    """
    Result from one crack detection pass.

    Args:
        crack_count: Number of crack-like contours retained after filtering.
        severity_score: Normalized 0-1 severity score.
        total_length_px: Sum of crack contour lengths in pixels.
        crack_area_px: Total crack mask area in pixels.
        boxes: Bounding boxes for detected cracks.
        mask: Binary crack mask.
        overlay: BGR image with cracks drawn over the source.
        warnings: Non-fatal quality or processing warnings.
    """

    crack_count: int
    severity_score: float
    total_length_px: float
    crack_area_px: int
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    mask: np.ndarray | None = None
    overlay: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly result payload."""
        return {
            "crack_count": self.crack_count,
            "severity_score": self.severity_score,
            "severity": self.severity_score,
            "total_length_px": self.total_length_px,
            "crack_area_px": self.crack_area_px,
            "boxes": self.boxes,
            "is_treatable": self.severity_score < CRACK_SEVERITY_THRESHOLD,
            "warnings": self.warnings,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style compatibility for older spike scripts."""
        return self.to_dict().get(key, default)


class CrackDetector:
    """
    OpenCV crack detector tuned by config.py thresholds.

    Args:
        min_area: Minimum contour area to keep.
        min_length: Minimum contour arc length to keep.
        min_aspect_ratio: Thinness filter for crack-like shapes.
        use_auto_canny: Whether to derive Canny thresholds from image median.
    """

    def __init__(
        self,
        *,
        min_area: int = MIN_CRACK_AREA,
        min_length: int = MIN_CRACK_LEN,
        min_aspect_ratio: float = MIN_ASPECT_RATIO,
        use_auto_canny: bool = USE_AUTO_CANNY,
    ) -> None:
        self.min_area = min_area
        self.min_length = min_length
        self.min_aspect_ratio = min_aspect_ratio
        self.use_auto_canny = use_auto_canny

    def detect(self, image: np.ndarray) -> CrackResult:
        """
        Detect crack-like dark line structures in a BGR or grayscale image.

        Args:
            image: OpenCV image array from camera or disk.

        Returns:
            CrackResult with masks, overlay, and scalar severity.
        """
        if image is None or image.size == 0:
            raise ValueError("image must be a non-empty numpy array")

        gray = self._to_gray(image)
        enhanced = self._enhance_contrast(gray)
        edges = self._edge_mask(enhanced)
        dark_lines = self._dark_line_mask(enhanced)
        # Dark-line segmentation carries the crack body; Canny contributes
        # crisp boundaries for very thin cracks. Requiring both can erase
        # valid hairline cracks on low-contrast stone.
        mask = cv2.bitwise_or(dark_lines, cv2.bitwise_and(edges, dark_lines))
        mask = self._clean_mask(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        crack_mask = np.zeros_like(mask)
        boxes: list[tuple[int, int, int, int]] = []
        total_length = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            length = cv2.arcLength(contour, closed=False)
            if area < self.min_area or length < self.min_length:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect = max(w, h) / max(1, min(w, h))
            slenderness = (length * length) / max(1.0, area)
            if aspect < self.min_aspect_ratio and slenderness < 80.0:
                continue

            boxes.append((x, y, w, h))
            total_length += length
            cv2.drawContours(crack_mask, [contour], -1, 255, thickness=cv2.FILLED)

        crack_area = int(np.count_nonzero(crack_mask))
        severity = self._score_severity(crack_area, total_length, gray.shape)
        overlay = self.draw_overlay(image, crack_mask, boxes, severity)

        result = CrackResult(
            crack_count=len(boxes),
            severity_score=severity,
            total_length_px=round(total_length, 2),
            crack_area_px=crack_area,
            boxes=boxes,
            mask=crack_mask,
            overlay=overlay,
        )
        logger.info(
            f"Crack detection: {result.crack_count} cracks, "
            f"severity={result.severity_score:.3f}"
        )
        return result

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Convert BGR/RGB/grayscale input to grayscale uint8."""
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"unsupported image shape: {image.shape}")
        return gray.astype(np.uint8, copy=False)

    @staticmethod
    def _enhance_contrast(gray: np.ndarray) -> np.ndarray:
        """Reduce illumination variation and emphasize fine dark cracks."""
        blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(blurred)

    def _edge_mask(self, gray: np.ndarray) -> np.ndarray:
        """Build an edge mask using either configured or automatic Canny."""
        if self.use_auto_canny:
            median = float(np.median(gray))
            lower = int(max(0, 0.66 * median))
            upper = int(min(255, 1.33 * median))
        else:
            lower, upper = CANNY_T1, CANNY_T2
        edges = cv2.Canny(gray, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.dilate(edges, kernel, iterations=1)

    @staticmethod
    def _dark_line_mask(gray: np.ndarray) -> np.ndarray:
        """Segment dark linework while adapting to uneven stone lighting."""
        blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, blackhat_kernel)
        _threshold, mask = cv2.threshold(
            blackhat,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return mask

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        """Remove isolated noise while preserving thin crack structures."""
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        return cleaned

    @staticmethod
    def _score_severity(area_px: int, length_px: float, shape: tuple[int, int]) -> float:
        """Compute normalized severity from crack area and length density."""
        height, width = shape
        image_area = max(1, height * width)
        diagonal = float(np.hypot(width, height))
        area_density = area_px / image_area
        length_density = length_px / max(1.0, diagonal)
        score = (area_density * 15.0) + (length_density * 0.35)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def draw_overlay(
        image: np.ndarray,
        mask: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
        severity: float,
    ) -> np.ndarray:
        """
        Draw crack mask and boxes over the input image.

        Args:
            image: Original BGR or grayscale image.
            mask: Binary crack mask.
            boxes: Crack bounding boxes.
            severity: Severity score for label color.

        Returns:
            BGR overlay image.
        """
        if image.ndim == 2:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            overlay = image.copy()

        crack_colour = np.zeros_like(overlay)
        crack_colour[:, :, 2] = mask
        overlay = cv2.addWeighted(overlay, 0.82, crack_colour, 0.55, 0)

        color = (0, 0, 255) if severity >= CRACK_SEVERITY_THRESHOLD else (0, 180, 255)
        for x, y, w, h in boxes:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
        cv2.putText(
            overlay,
            f"Crack severity: {severity:.2f}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )
        return overlay


def run_crack_detection(image: np.ndarray) -> CrackResult:
    """
    Convenience function for the mission pipeline.

    Args:
        image: BGR or grayscale OpenCV image.

    Returns:
        CrackResult from a default CrackDetector.
    """
    return CrackDetector().detect(image)


def _synthetic_test_image() -> np.ndarray:
    """Create a synthetic stone tile with several cracks for smoke tests."""
    image = np.full((360, 520, 3), (150, 142, 122), dtype=np.uint8)
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 8, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.line(image, (90, 120), (210, 170), (28, 28, 28), 3)
    cv2.line(image, (210, 170), (390, 150), (22, 22, 22), 2)
    cv2.line(image, (245, 168), (270, 235), (30, 30, 30), 2)
    return image


if __name__ == "__main__":
    output_dir = Path("data/test_crack_detector")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _synthetic_test_image()
    result = run_crack_detection(frame)
    cv2.imwrite(str(output_dir / "synthetic_cracks.png"), frame)
    cv2.imwrite(str(output_dir / "synthetic_crack_overlay.png"), result.overlay)
    print("AXUM Crack Detector synthetic test")
    print(result.to_dict())
    print(f"Saved overlay to: {output_dir / 'synthetic_crack_overlay.png'}")
