"""
AXUM ROVER - Photometric Stereo Relief Mapping
==============================================
WHAT: Reconstructs relative surface normals and relief from four quadrant-lit
frames.
WHY: Inscriptions and tool marks need visual relief maps even when absolute
metric depth is not calibrated.
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
    PHOTOMETRIC_ALBEDO_EPSILON,
    PHOTOMETRIC_DEPTH_GRADIENT_SCALE,
    PHOTOMETRIC_DEPTH_SMOOTH_ITERATIONS,
    SCAN_PHOTOS_DIR,
)


# SECTION 1 - DATA STRUCTURES

@dataclass
class PhotometricStereoResult:
    """
    WHAT: Complete relative photometric-stereo output.
    WHY: Callers must not mistake this uncalibrated relief map for metric depth.
    """

    normal_map: np.ndarray
    relative_depth_map: np.ndarray
    albedo_map: np.ndarray
    quality_score: float
    light_directions: dict[str, tuple[float, float, float]]
    warnings: list[str] = field(default_factory=list)
    is_relative_depth: bool = True
    depth_units: str = "relative_uncalibrated"

    def to_dict(self) -> dict[str, Any]:
        """WHAT: Serialise scalar metadata. WHY: Numpy arrays are saved separately."""
        return {
            "quality_score": self.quality_score,
            "light_directions": self.light_directions,
            "warnings": self.warnings,
            "is_relative_depth": self.is_relative_depth,
            "depth_units": self.depth_units,
        }


# SECTION 2 - HARDWARE CAPTURE

def load_quadrant_frames(paths: dict[str, str | Path]) -> dict[str, np.ndarray]:
    """
    WHAT: Load already captured N/E/S/W quadrant frames.
    WHY: This module consumes frames; serial LED timing is firmware/pipeline work.
    """
    frames: dict[str, np.ndarray] = {}
    for key in ("N", "E", "S", "W"):
        if key not in paths:
            raise KeyError(f"missing quadrant frame: {key}")
        frame = cv2.imread(str(paths[key]), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"quadrant frame not readable: {paths[key]}")
        frames[key] = frame
    return frames


# SECTION 3 - CORE MATH

def default_light_directions() -> dict[str, tuple[float, float, float]]:
    """
    WHAT: Return fixed approximate quadrant LED directions.
    WHY: Relative relief reconstruction needs known lighting geometry.
    """
    z = 0.8
    xy = 0.6
    dirs = {
        "N": (0.0, -xy, z),
        "E": (xy, 0.0, z),
        "S": (0.0, xy, z),
        "W": (-xy, 0.0, z),
    }
    return {k: tuple((np.array(v) / np.linalg.norm(v)).tolist()) for k, v in dirs.items()}


def _gray_float(frame: np.ndarray) -> np.ndarray:
    """WHAT: Convert image to grayscale float. WHY: Lambertian solve uses intensity."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        raise ValueError(f"unsupported frame shape: {frame.shape}")
    return gray.astype(np.float32) / 255.0


def solve_normals(
    quadrant_frames: dict[str, np.ndarray],
    light_directions: dict[str, tuple[float, float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    WHAT: Solve per-pixel Lambertian normals from four lit images.
    WHY: Overdetermined least-squares uses all quadrants and stays CPU-friendly.
    """
    if light_directions is None:
        light_directions = default_light_directions()
    keys = ["N", "E", "S", "W"]
    intensities = [_gray_float(quadrant_frames[k]) for k in keys]
    shape = intensities[0].shape
    if any(img.shape != shape for img in intensities):
        raise ValueError("quadrant frames must share one pixel grid")

    light_matrix = np.array([light_directions[k] for k in keys], dtype=np.float32)
    pseudo_inv = np.linalg.pinv(light_matrix)
    stacked = np.stack(intensities, axis=0).reshape(4, -1)
    g = pseudo_inv @ stacked
    albedo = np.linalg.norm(g, axis=0)
    normals = g / (albedo + PHOTOMETRIC_ALBEDO_EPSILON)
    normal_map = normals.T.reshape(shape[0], shape[1], 3).astype(np.float32)
    albedo_map = albedo.reshape(shape).astype(np.float32)
    return normal_map, albedo_map


def integrate_normals_to_relative_depth(normal_map: np.ndarray) -> np.ndarray:
    """
    WHAT: Integrate normals into a smoothed relative relief map.
    WHY: The output is for visualization, not calibrated absolute metrology.
    """
    nz = np.clip(normal_map[:, :, 2], 0.1, None)
    p = -normal_map[:, :, 0] / nz * PHOTOMETRIC_DEPTH_GRADIENT_SCALE
    q = -normal_map[:, :, 1] / nz * PHOTOMETRIC_DEPTH_GRADIENT_SCALE
    depth = np.zeros(p.shape, dtype=np.float32)
    for _ in range(PHOTOMETRIC_DEPTH_SMOOTH_ITERATIONS):
        depth[1:, :] += q[:-1, :] * 0.25
        depth[:, 1:] += p[:, :-1] * 0.25
        depth = cv2.GaussianBlur(depth, (3, 3), 0)
        depth -= float(depth.mean())
    denom = float(np.max(np.abs(depth)))
    if denom > 0:
        depth = depth / denom
    return depth.astype(np.float32)


def run_photometric_stereo(
    quadrant_frames: dict[str, np.ndarray],
    light_directions: dict[str, tuple[float, float, float]] | None = None,
) -> PhotometricStereoResult:
    """
    WHAT: Reconstruct relative normals and relief from N/E/S/W frames.
    WHY: Provides inscription/tool-mark visualization without requiring GPU.
    """
    if light_directions is None:
        light_directions = default_light_directions()
    normal_map, albedo_map = solve_normals(quadrant_frames, light_directions)
    depth_map = integrate_normals_to_relative_depth(normal_map)
    quality = float(np.clip(np.mean(albedo_map > 0.03), 0.0, 1.0))
    result = PhotometricStereoResult(
        normal_map=normal_map,
        relative_depth_map=depth_map,
        albedo_map=albedo_map,
        quality_score=quality,
        light_directions=light_directions,
    )
    logger.info(f"Photometric stereo: quality={quality:.3f}")
    return result


# SECTION 4 - VISUALISATION

def visualise_normal_map(normal_map: np.ndarray) -> np.ndarray:
    """
    WHAT: Convert normals from [-1, 1] to BGR display colors.
    WHY: Normal maps make relief direction visible to operators.
    """
    rgb = ((normal_map + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def visualise_depth_map(relative_depth_map: np.ndarray) -> np.ndarray:
    """
    WHAT: Convert relative depth to a false-color heatmap.
    WHY: Relief maps need compact visual QA and dashboard display.
    """
    scaled = ((relative_depth_map + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)


# SECTION 5 - MAIN ENTRY POINT

def analyse_photometric_paths(paths: dict[str, str | Path]) -> PhotometricStereoResult:
    """
    WHAT: Load quadrant frames and run photometric stereo.
    WHY: Provides a simple pipeline entry point for saved scan folders.
    """
    return run_photometric_stereo(load_quadrant_frames(paths))


# SECTION 6 - CALIBRATION UTILITY

def calibrate_light_balance(quadrant_frames: dict[str, np.ndarray]) -> dict[str, float]:
    """
    WHAT: Estimate per-quadrant brightness balance.
    WHY: Unequal LED brightness should be measured before interpreting relief.
    """
    means = {k: float(_gray_float(v).mean()) for k, v in quadrant_frames.items()}
    avg = float(np.mean(list(means.values()))) if means else 1.0
    return {k: (avg / max(v, PHOTOMETRIC_ALBEDO_EPSILON)) for k, v in means.items()}


# SECTION 7 - SYNTHETIC __main__ TEST

def _synthetic_quadrant_frames() -> dict[str, np.ndarray]:
    """WHAT: Generate shaded relief frames. WHY: Standalone test must need no hardware."""
    h, w = 220, 320
    y, x = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
    z = np.exp(-((x * 2.2) ** 2 + (y * 2.8) ** 2)) * 0.35
    z += np.exp(-(((x + 0.35) * 18) ** 2 + ((y - 0.05) * 4) ** 2)) * 0.12
    gy, gx = np.gradient(z)
    normals = np.dstack((-gx, -gy, np.ones_like(z)))
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    frames: dict[str, np.ndarray] = {}
    for key, light in default_light_directions().items():
        lvec = np.array(light, dtype=np.float32)
        intensity = np.clip((normals @ lvec) * 0.8 + 0.15, 0.0, 1.0)
        frame = (intensity * 255).astype(np.uint8)
        frames[key] = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frames


if __name__ == "__main__":
    out_dir = SCAN_PHOTOS_DIR / "_synthetic_photometric"
    out_dir.mkdir(parents=True, exist_ok=True)
    quad_frames = _synthetic_quadrant_frames()
    ps_result = run_photometric_stereo(quad_frames)
    cv2.imwrite(str(out_dir / "normal_map.png"), visualise_normal_map(ps_result.normal_map))
    cv2.imwrite(str(out_dir / "relative_depth.png"), visualise_depth_map(ps_result.relative_depth_map))
    for name, frame in quad_frames.items():
        cv2.imwrite(str(out_dir / f"quad_{name}.png"), frame)
    print("AXUM Photometric Stereo synthetic test")
    print(ps_result.to_dict())
    print(f"Saved outputs to: {out_dir}")
