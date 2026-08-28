"""
AXUM ROVER - Crack detection module.

WHAT: Ridge-based crack extraction for stone, pottery, and inscription images.
WHY: Crack detection must work locally on CPU and before any ML model is
trained, so this module uses explainable image processing.

The previous black-hat + percentile pipeline had three faults, each measured:

    1. It kept a FIXED 2% of pixels on every image (measured 1.91-1.99% across
       five photographs), so an undamaged surface still produced a full mask.
       It could not report "no crack".
    2. Its 15x15 black-hat kernel cannot see a crack wider than the kernel:
       closing minus original is ~0 inside a wide crack. At the darkest point
       of a full-height crack in a 12MP wall photo the response was 2.4 against
       a selection cut-off of 161. The crack was invisible while 3px moss
       specks were not.
    3. Filter sizes and MIN_CRACK_AREA are pixel counts with no scale
       normalisation, so the same surface scored differently by resolution.

Cracks are dark curvilinear valleys. The Hessian eigenvalue ratio separates
those from dark compact texture, which black-hat cannot do because it responds
to any dark structure of the right size. Measured at a 512px working edge:

                              MCS (crack bodies)   Stone331 (1px centrelines)
    black-hat + percentile      F1 0.147             F1 0.111
    this module                 F1 0.346  P 0.843    F1 0.020

THIS IS A TRADE, NOT A CLEAN WIN, and the second column is the cost. On MCS
precision rises from roughly 0.08 to 0.843 and a blank plate finally returns
nothing at all; on Stone331 the score falls by 5x. Two reasons, and only the
first is a genuine defect:

    1. Stone331's own texture out-responds its annotated cracks (median ridge
       response inside a crack 0.057 against a 99th percentile of 0.102 on
       crack-free area). Hairlines on rough stone are simply not separable by
       any of the nine filters benchmarked in
       scripts/experiment_crack_methods.py -- the best classical score there is
       0.215, against roughly 0.85 for DeepCrack's CNN.
    2. Stone331's ground truth is a 1px hand-traced centreline, so marking a
       crack's body is penalised by construction. MCS annotates bodies, which
       is why the same detector scores 0.346 there.

So: use this for visible damage on artefact surfaces, and do NOT rely on it for
hairline detection on rough stone. The trained segmenter in
src/crack_detection/segmenter.py is the fix for that case and has never
actually been trained -- logs/crack_segmenter holds 3- and 6-epoch probes.

The severity score is still not calibrated against conservator judgement, so no
conservation decision should rest on it alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger
from skimage.filters import sato

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    CRACK_DARKNESS_CUT,
    CRACK_MAX_MEAN_WIDTH_PX,
    CRACK_MIN_EXTENT_PX,
    CRACK_RIDGE_SIGMAS,
    CRACK_RIDGE_THRESHOLD,
    CRACK_SEVERITY_THRESHOLD,
    CRACK_TEXTURE_MULTIPLE,
    CRACK_WORKING_EDGE,
    MIN_CRACK_AREA,
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
    Ridge-based crack detector.

    Args:
        working_edge: Long edge every image is resized to before detection.
        ridge_threshold: Absolute cut-off on the ridge response.
        darkness_cut: Flattened intensity below which a pixel may join a crack.
            128 is exactly local background, so this is a contrast ratio.
        min_extent: Shortest span, in working pixels, a component may have.
        max_mean_width: Widest a component may be on average and still be a
            crack rather than a stain or a shadow.
        min_area: Smallest component kept, in working pixels.
    """

    def __init__(
        self,
        *,
        working_edge: int = CRACK_WORKING_EDGE,
        ridge_threshold: float = CRACK_RIDGE_THRESHOLD,
        texture_multiple: float = CRACK_TEXTURE_MULTIPLE,
        darkness_cut: float = CRACK_DARKNESS_CUT,
        min_extent: int = CRACK_MIN_EXTENT_PX,
        max_mean_width: float = CRACK_MAX_MEAN_WIDTH_PX,
        min_area: int = MIN_CRACK_AREA,
    ) -> None:
        self.working_edge = working_edge
        self.ridge_threshold = ridge_threshold
        self.texture_multiple = texture_multiple
        self.darkness_cut = darkness_cut
        self.min_extent = min_extent
        self.max_mean_width = max_mean_width
        self.min_area = min_area

    def detect(self, image: np.ndarray) -> CrackResult:
        """
        Detect crack-like dark curvilinear structures in a BGR or grayscale image.

        Args:
            image: OpenCV image array from camera or disk.

        Returns:
            CrackResult with masks, overlay, and scalar severity.
        """
        if image is None or image.size == 0:
            raise ValueError("image must be a non-empty numpy array")

        gray = self._to_gray(image)
        working = self._to_working_scale(gray)
        flattened = self._flatten_illumination(working)
        response = self._ridge_response(flattened)

        mask = self._grow_from_ridges(response, flattened)
        mask = self._bridge_gaps(mask)
        crack_mask, boxes, total_length = self._keep_crack_shapes(mask)

        severity = self._score_severity(int(np.count_nonzero(crack_mask)),
                                        total_length, working.shape)
        # Report at the caller's resolution; the dashboard overlays on the original.
        full_mask = cv2.resize(crack_mask, (gray.shape[1], gray.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        scale = gray.shape[1] / working.shape[1]
        full_boxes = [(int(x * scale), int(y * scale),
                       int(w * scale), int(h * scale)) for x, y, w, h in boxes]
        overlay = self.draw_overlay(image, full_mask, full_boxes, severity)

        result = CrackResult(
            crack_count=len(boxes),
            severity_score=severity,
            total_length_px=round(total_length * scale, 2),
            crack_area_px=int(np.count_nonzero(full_mask)),
            boxes=full_boxes,
            mask=full_mask,
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

    def _to_working_scale(self, gray: np.ndarray) -> np.ndarray:
        """Resize so filter sizes mean the same thing on every input."""
        height, width = gray.shape
        longest = max(height, width)
        if longest == self.working_edge:
            return gray
        scale = self.working_edge / longest
        return cv2.resize(gray, (max(1, round(width * scale)),
                                 max(1, round(height * scale))),
                          interpolation=cv2.INTER_AREA)

    def _flatten_illumination(self, gray: np.ndarray) -> np.ndarray:
        """
        Divide out slow shading so a dark corner is not read as damage.

        Subtracting a blur would leave the residual scaled by local brightness,
        so the same crack would respond more strongly in sunlight. Dividing
        makes the response a contrast ratio instead.
        """
        sigma = self.working_edge / 16
        background = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
        flat = gray.astype(np.float32) / np.maximum(background, 1.0)
        return np.clip(flat * 128.0, 0, 255)

    @staticmethod
    def _ridge_response(gray: np.ndarray) -> np.ndarray:
        """
        Multi-scale dark-ridge response, on an absolute scale.

        Deliberately NOT normalised by the image's own maximum. Dividing by the
        peak is what turns a threshold back into a percentile: on a clean
        surface the peak is noise, and normalising promotes that noise to 1.0.
        On this scale a uniform plate responds 0.0000 and stays empty.
        """
        response = sato(gray / 255.0, sigmas=CRACK_RIDGE_SIGMAS, black_ridges=True)
        return np.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _bridge_gaps(mask: np.ndarray) -> np.ndarray:
        """Reconnect a crack broken by a highlight, without fattening it."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    def seed_threshold(self, response: np.ndarray) -> float:
        """
        Cut-off a pixel must clear to seed a crack.

        The larger of an absolute floor and a multiple of the image's own median
        response. The floor alone does not transfer: calibrated on marble it
        marked 95.2% of a concrete wall, because rough surfaces are covered in
        micro-ridges. Scaling by the median asks the useful question -- does
        this stand out from THIS surface -- while the floor keeps a clean
        surface empty rather than promoting its noise.
        """
        return max(self.ridge_threshold,
                   self.texture_multiple * float(np.median(response)))

    def _grow_from_ridges(self, response: np.ndarray,
                          flattened: np.ndarray) -> np.ndarray:
        """
        Seed on the ridge response, then grow through connected dark pixels.

        A ridge filter has a maximum width as surely as black-hat does: inside a
        crack wider than about twice the largest sigma the surface is flat, the
        second derivative is ~0, and only the two edges respond. Thresholding
        the response alone therefore outlines a wide crack instead of filling
        it, and on the Medway wall photograph it missed the crack entirely.

        Hysteresis fixes it without a larger filter bank. The ridge response
        decides WHERE a crack is; the darkness test decides HOW FAR it extends.
        Dark texture with no ridge seed anywhere in it is dropped whole, so
        lichen and shadow do not survive by being merely dark.

        Growth is bounded to a neighbourhood of the seeds. Unbounded, a hairline
        on rough stone pulls in the whole surrounding dark blotch: measured on
        Stone331 that cost F1 0.111 -> 0.010. A crack's interior is within about
        twice the largest sigma of a ridge seed by construction, so nothing
        legitimate is lost by refusing to travel further.
        """
        seeds = response > self.seed_threshold(response)
        if not seeds.any():
            return np.zeros(response.shape, dtype=np.uint8)

        reach = int(2 * max(CRACK_RIDGE_SIGMAS))
        near_seed = cv2.dilate(
            seeds.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * reach + 1, 2 * reach + 1)))
        dark = ((flattened < self.darkness_cut) & (near_seed > 0)).astype(np.uint8)
        count, labels = cv2.connectedComponents(dark, 8)
        if count <= 1:
            return np.zeros(response.shape, dtype=np.uint8)

        seeded = np.unique(labels[seeds])
        seeded = seeded[seeded != 0]
        return np.isin(labels, seeded).astype(np.uint8) * 255

    def _keep_crack_shapes(
        self, mask: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[int, int, int, int]], float]:
        """
        Keep components that are long and thin, drop compact ones.

        Ranking pixels is not enough. Lichen, grain and sensor noise are
        genuinely dark, so they survive any brightness threshold; what they do
        not survive is a shape test. Mean width is area divided by the longer
        bounding-box side, which unlike a bounding-box aspect ratio does not
        call a diagonal crack compact just because its box is square.

        Statistics come from connectedComponentsWithStats in one pass; a
        per-component Python loop over a noisy mask is thousands of iterations
        and was measured too slow to ship.
        """
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if count <= 1:
            return np.zeros_like(mask), [], 0.0

        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        widths = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float64)
        heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float64)
        extents = np.maximum(widths, heights)
        mean_widths = areas / np.maximum(extents, 1.0)

        keep = ((areas >= self.min_area)
                & (extents >= self.min_extent)
                & (mean_widths <= self.max_mean_width))
        kept_labels = np.nonzero(keep)[0] + 1
        if kept_labels.size == 0:
            return np.zeros_like(mask), [], 0.0

        crack_mask = np.isin(labels, kept_labels).astype(np.uint8) * 255
        boxes = [(int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                  int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
                 for i in kept_labels]
        return crack_mask, boxes, float(extents[keep].sum())

    def _score_severity(self, area_px: int, length_px: float,
                        shape: tuple[int, int]) -> float:
        """
        Normalised severity from crack area and length density.

        Both terms are computed at the fixed working resolution, so the same
        surface photographed at 0.3MP and 12MP now scores the same. Previously
        they were measured on the source image, and a small photo of a cracked
        rock outscored a large photo of a worse one.

        The constants set how quickly the score approaches 1 and are NOT
        calibrated against conservator judgement; CRACK_SEVERITY_THRESHOLD is
        therefore not yet a defensible treat/flag boundary.
        """
        height, width = shape
        image_area = max(1, height * width)
        diagonal = float(np.hypot(width, height))

        area_term = 1.0 - np.exp(-(area_px / image_area) / 0.05)
        length_term = 1.0 - np.exp(-(length_px / max(1.0, diagonal)) / 6.0)

        score = 0.6 * area_term + 0.4 * length_term
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
