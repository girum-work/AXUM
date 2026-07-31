"""
AXUM ROVER - Multispectral Stress Mapping
=========================================
WHAT: Computes visible/IR ratiometric contrast maps for crack-associated
surface stress.
WHY: The mission pipeline needs a CPU-only scalar stress signal from already
captured visible/IR pairs without duplicating crack geometry detection.
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
    MULTISPECTRAL_EPSILON,
    MULTISPECTRAL_HAZARD_SCORE_MIN,
    MULTISPECTRAL_MIN_REGION_AREA_PX,
    MULTISPECTRAL_SURFACE_ONLY_SCORE_MAX,
    SCAN_PHOTOS_DIR,
)
from src.crack_detection.detector import CrackDetector, CrackResult


# SECTION 1 - DATA STRUCTURES

@dataclass
class MultispectralRegion:
    """
    WHAT: One crack region scored from the NDCI map.
    WHY: Region-level scores let the treatment/fragility layers distinguish
    structural hazards from visible but low-contrast surface marks.
    """

    bbox: tuple[int, int, int, int]
    area_px: int
    ndci_mean: float
    ndci_auc: float
    hazard_score: float
    classification: str

    def to_dict(self) -> dict[str, Any]:
        """WHAT: Return JSON-friendly region data. WHY: Dashboard events need plain types."""
        return self.__dict__.copy()


@dataclass
class MultispectralResult:
    """
    WHAT: Complete visible/IR stress analysis output.
    WHY: Keeps scalar fragility inputs and visual products together for callers.
    """

    visible_image: np.ndarray
    ir_image: np.ndarray
    ndci_map: np.ndarray
    crack_mask: np.ndarray
    stress_map: np.ndarray
    regions: list[MultispectralRegion] = field(default_factory=list)
    structural_hazard_score: float = 0.0
    surface_only_score: float = 0.0
    quality_score: float = 1.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """WHAT: Serialise scalar fields. WHY: Numpy arrays are not JSON payloads."""
        return {
            "regions": [r.to_dict() for r in self.regions],
            "structural_hazard_score": self.structural_hazard_score,
            "surface_only_score": self.surface_only_score,
            "quality_score": self.quality_score,
            "warnings": self.warnings,
        }


# SECTION 2 - HARDWARE CAPTURE

def load_aligned_frame_pair(
    visible_path: str | Path,
    ir_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    WHAT: Load an already captured visible/IR pair from disk.
    WHY: LED sequencing belongs to firmware; this module consumes aligned images only.
    """
    visible = cv2.imread(str(visible_path), cv2.IMREAD_COLOR)
    ir = cv2.imread(str(ir_path), cv2.IMREAD_COLOR)
    if visible is None:
        raise FileNotFoundError(f"visible frame not readable: {visible_path}")
    if ir is None:
        raise FileNotFoundError(f"IR frame not readable: {ir_path}")
    return visible, ir


# SECTION 3 - CORE MATH

def _to_gray_float(image: np.ndarray) -> np.ndarray:
    """WHAT: Convert BGR/grayscale image to float intensity. WHY: NDCI is scalar per pixel."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError(f"unsupported image shape: {image.shape}")
    return gray.astype(np.float32) / 255.0


def compute_ndci_map(visible_image: np.ndarray, ir_image: np.ndarray) -> np.ndarray:
    """
    WHAT: Compute NDCI = (visible - IR) / (visible + IR) per pixel.
    WHY: Ratiometric contrast suppresses illumination changes while exposing
    crack-associated spectral stress.
    """
    vis = _to_gray_float(visible_image)
    ir = _to_gray_float(ir_image)
    if vis.shape != ir.shape:
        raise ValueError(f"visible/IR frames must be aligned; got {vis.shape} vs {ir.shape}")
    ndci = (vis - ir) / (vis + ir + MULTISPECTRAL_EPSILON)
    return np.clip(ndci, -1.0, 1.0).astype(np.float32)


def score_crack_regions(
    ndci_map: np.ndarray,
    crack_result: CrackResult,
) -> list[MultispectralRegion]:
    """
    WHAT: Score crack contours using the provided crack detector mask.
    WHY: Crack geometry has one owner, so this module only measures spectral
    contrast inside those existing regions.
    """
    if crack_result.mask is None:
        return []
    mask = cv2.resize(
        crack_result.mask,
        (ndci_map.shape[1], ndci_map.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    regions: list[MultispectralRegion] = []

    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < MULTISPECTRAL_MIN_REGION_AREA_PX:
            continue
        region_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, 255, thickness=cv2.FILLED)
        values = np.abs(ndci_map[region_mask > 0])
        if values.size == 0:
            continue
        ndci_mean = float(values.mean())
        ndci_auc = float(values.sum() / max(1, ndci_map.size))
        hazard_score = float(np.clip(
            ndci_mean * min(1.0, area / 500.0) + ndci_auc * 4.0,
            0.0,
            1.0,
        ))
        if hazard_score >= MULTISPECTRAL_HAZARD_SCORE_MIN:
            classification = "hazard"
        elif hazard_score <= MULTISPECTRAL_SURFACE_ONLY_SCORE_MAX:
            classification = "surface-only"
        else:
            classification = "monitor"
        regions.append(MultispectralRegion(
            cv2.boundingRect(contour),
            area,
            ndci_mean,
            ndci_auc,
            hazard_score,
            classification,
        ))

    regions.sort(key=lambda r: r.hazard_score, reverse=True)
    return regions


def run_multispectral(
    visible_image: np.ndarray,
    ir_image: np.ndarray,
    crack_result: CrackResult | None = None,
) -> MultispectralResult:
    """
    WHAT: Run CPU-only multispectral stress mapping on an aligned frame pair.
    WHY: Produces the stress score consumed by fragility and treatment modules.
    """
    if crack_result is None:
        crack_result = CrackDetector().detect(visible_image)

    ndci = compute_ndci_map(visible_image, ir_image)
    regions = score_crack_regions(ndci, crack_result)
    crack_mask = (
        crack_result.mask
        if crack_result.mask is not None
        else np.zeros(ndci.shape, dtype=np.uint8)
    )
    if crack_mask.shape != ndci.shape:
        crack_mask = cv2.resize(
            crack_mask,
            (ndci.shape[1], ndci.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    stress_map = (np.abs(ndci) * (crack_mask > 0)).astype(np.float32)

    hazard_scores = [r.hazard_score for r in regions if r.classification == "hazard"]
    surface_scores = [
        r.hazard_score for r in regions if r.classification == "surface-only"
    ]
    result = MultispectralResult(
        visible_image=visible_image,
        ir_image=ir_image,
        ndci_map=ndci,
        crack_mask=crack_mask,
        stress_map=stress_map,
        regions=regions,
        structural_hazard_score=max(hazard_scores, default=0.0),
        surface_only_score=max(surface_scores, default=0.0),
    )
    logger.info(
        f"Multispectral: regions={len(regions)}, hazard={result.structural_hazard_score:.3f}"
    )
    return result


# SECTION 4 - VISUALISATION

def visualise_multispectral_result(result: MultispectralResult) -> np.ndarray:
    """
    WHAT: Draw NDCI heatmap and crack region boxes over the visible frame.
    WHY: Operators need to see where spectral stress came from.
    """
    heat = ((result.ndci_map + 1.0) * 127.5).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(result.visible_image, 0.65, heat_bgr, 0.35, 0)
    for region in result.regions:
        x, y, w, h = region.bbox
        color = (0, 0, 255) if region.classification == "hazard" else (0, 180, 255)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
    return overlay


# SECTION 5 - MAIN ENTRY POINT

def analyse_multispectral_paths(
    visible_path: str | Path,
    ir_path: str | Path,
) -> MultispectralResult:
    """
    WHAT: Load frame paths and run multispectral analysis.
    WHY: Provides a compact entry point for scripts and pipeline wiring.
    """
    visible, ir = load_aligned_frame_pair(visible_path, ir_path)
    return run_multispectral(visible, ir)


# SECTION 6 - CALIBRATION UTILITY

def calibrate_ndci_baseline(
    frame_pairs: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    """
    WHAT: Estimate baseline NDCI distribution from known-stable frame pairs.
    WHY: Real camera/LED response can bias the ratio and should be documented.
    """
    values = []
    for visible, ir in frame_pairs:
        values.append(compute_ndci_map(visible, ir).ravel())
    if not values:
        return {"mean": 0.0, "std": 0.0}
    merged = np.concatenate(values)
    return {"mean": float(merged.mean()), "std": float(merged.std())}


# SECTION 7 - SYNTHETIC __main__ TEST

def _synthetic_pair() -> tuple[np.ndarray, np.ndarray, CrackResult]:
    """WHAT: Build fake visible/IR frames. WHY: Standalone smoke tests need no hardware."""
    h, w = 220, 320
    visible = np.full((h, w, 3), 150, dtype=np.uint8)
    ir = np.full((h, w, 3), 145, dtype=np.uint8)
    cv2.line(visible, (45, 75), (145, 95), (230, 230, 230), 8)
    cv2.line(ir, (45, 75), (145, 95), (45, 45, 45), 8)
    cv2.line(visible, (190, 145), (270, 150), (165, 165, 165), 5)
    cv2.line(ir, (190, 145), (270, 150), (155, 155, 155), 5)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.line(mask, (45, 75), (145, 95), 255, 8)
    cv2.line(mask, (190, 145), (270, 150), 255, 5)
    crack = CrackResult(
        2,
        0.5,
        180.0,
        int(np.count_nonzero(mask)),
        [(45, 70, 105, 30), (190, 140, 85, 15)],
        mask,
    )
    return visible, ir, crack


if __name__ == "__main__":
    out_dir = SCAN_PHOTOS_DIR / "_synthetic_multispectral"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis, ir_frame, cracks = _synthetic_pair()
    ms_result = run_multispectral(vis, ir_frame, cracks)
    overlay = visualise_multispectral_result(ms_result)
    cv2.imwrite(str(out_dir / "visible.png"), vis)
    cv2.imwrite(str(out_dir / "ir.png"), ir_frame)
    cv2.imwrite(str(out_dir / "overlay.png"), overlay)
    print("AXUM Multispectral synthetic test")
    print(ms_result.to_dict())
    print(f"Saved overlay to: {out_dir / 'overlay.png'}")
