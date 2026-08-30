"""
marker_nav.py — ArUco marker sighting and docking guidance.

WHY THIS EXISTS: mission_tree.py's NAVIGATE phase has been an explicit stub
that refuses to report SUCCESS on hardware, because "ArUco docking was designed
on paper but never built". behavior_tree.py already consumes a
`navigate_confidence` float and branches on it -- below `abort_below` is a hard
failure, between that and `retry_below` is worth one more attempt. This module
produces that number.

WHAT IT WILL AND WILL NOT TELL YOU. Detection needs nothing but the image.
Distance needs the camera's focal length and the marker's physical width, and
getting either wrong scales every range by the same factor. So:

    no calibration    marker id, image position, and a bearing that assumes a
                      nominal field of view. Confidence is capped at
                      UNCALIBRATED_CEILING, because an assumed FOV is a guess
                      and the rover should not drive confidently on a guess.
    calibrated        full pose by IPPE_SQUARE, which solves the planar-square
                      case directly rather than treating four coplanar points
                      as a general PnP problem. Range and marker yaw become
                      meaningful.

Run scripts/calibrate_camera.py to move from the first case to the second.

CONFIDENCE IS DELIBERATELY PESSIMISTIC. It falls with small markers, with
oblique views, and with the calibration's own reprojection error. A number that
flatters itself here does not cause a bad reading, it causes the behaviour tree
to skip the retry that would have fixed one.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

# Matches the markers already generated for this project and the dictionary
# pi/tools/pi_aruco_benchmark.py profiles against.
DEFAULT_DICTIONARY = cv2.aruco.DICT_6X6_250
# Bearing from an assumed field of view is a guess; scale down what it may claim.
UNCALIBRATED_CEILING = 0.40
# Below this share of the frame a marker's corners are too few pixels apart for
# the pose to be steady, whatever the solver reports.
MIN_AREA_FRACTION = 0.0004


@dataclass
class MarkerSighting:
    """One marker seen in one frame."""

    marker_id: int
    centre_px: tuple[float, float]
    area_px: float
    bearing_deg: float
    confidence: float
    range_m: float | None = None
    yaw_deg: float | None = None
    calibrated: bool = False
    corners: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"marker_id": self.marker_id,
                "centre_px": [round(v, 1) for v in self.centre_px],
                "bearing_deg": round(self.bearing_deg, 2),
                "range_m": None if self.range_m is None else round(self.range_m, 3),
                "yaw_deg": None if self.yaw_deg is None else round(self.yaw_deg, 2),
                "confidence": round(self.confidence, 3),
                "calibrated": self.calibrated}


@dataclass
class DockingCommand:
    """What to do about it, and how much to believe it."""

    turn_deg: float = 0.0
    advance_m: float | None = None
    confidence: float = 0.0
    reason: str = "no marker"
    sighting: MarkerSighting | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"turn_deg": round(self.turn_deg, 2),
                "advance_m": None if self.advance_m is None else round(self.advance_m, 3),
                "confidence": round(self.confidence, 3),
                "reason": self.reason,
                "warnings": self.warnings,
                "sighting": self.sighting.to_dict() if self.sighting else None}


class MarkerNavigator:
    """
    Args:
        calibration: JSON from scripts/calibrate_camera.py. Without it, range
            and yaw stay None and confidence is capped.
        marker_length_m: Printed marker edge, black border included. Measure
            it; a 5% error here is a 5% error in every range.
        nominal_hfov_deg: Only used when uncalibrated, and only for bearing.
    """

    def __init__(self, calibration: Path | None = None,
                 marker_length_m: float = 0.08,
                 dictionary: int = DEFAULT_DICTIONARY,
                 nominal_hfov_deg: float = 62.0) -> None:
        self.marker_length_m = marker_length_m
        self.nominal_hfov_deg = nominal_hfov_deg
        self.camera_matrix: np.ndarray | None = None
        self.distortion: np.ndarray | None = None
        self.reprojection_error: float | None = None

        if calibration and Path(calibration).exists():
            record = json.loads(Path(calibration).read_text(encoding="utf-8"))
            self.camera_matrix = np.array(record["camera_matrix"], dtype=np.float64)
            self.distortion = np.array(record["distortion"], dtype=np.float64)
            self.reprojection_error = float(record.get("reprojection_error_px", 0.0))
            self.nominal_hfov_deg = float(record.get("hfov_deg", nominal_hfov_deg))
            logger.info(f"Marker nav calibrated: reprojection "
                        f"{self.reprojection_error:.3f}px, "
                        f"hfov {self.nominal_hfov_deg:.1f} deg")
        else:
            logger.warning("Marker nav running UNCALIBRATED: bearing only, "
                           f"confidence capped at {UNCALIBRATED_CEILING}. "
                           "Run scripts/calibrate_camera.py.")

        self._dictionary = cv2.aruco.getPredefinedDictionary(dictionary)
        parameters = cv2.aruco.DetectorParameters()
        # Corners decide the pose, so refine them; the cost is small next to
        # the detection itself, measured at ~10ms on the Pi.
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(self._dictionary, parameters)

    @property
    def calibrated(self) -> bool:
        return self.camera_matrix is not None

    def detect(self, frame: np.ndarray) -> list[MarkerSighting]:
        """Every marker in the frame, strongest first."""
        if frame is None or frame.size == 0:
            return []
        gray = (frame if frame.ndim == 2
                else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        corner_sets, ids, _rejected = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return []

        height, width = gray.shape[:2]
        sightings = []
        for corners, marker_id in zip(corner_sets, ids.ravel()):
            sightings.append(self._measure(corners.reshape(4, 2),
                                           int(marker_id), width, height))
        return sorted(sightings, key=lambda s: s.confidence, reverse=True)

    def _measure(self, corners: np.ndarray, marker_id: int,
                 width: int, height: int) -> MarkerSighting:
        centre = (float(corners[:, 0].mean()), float(corners[:, 1].mean()))
        area = float(cv2.contourArea(corners.astype(np.float32)))
        area_fraction = area / max(1, width * height)

        # Positive bearing means the marker sits right of the optical axis.
        principal_x = (float(self.camera_matrix[0, 2]) if self.calibrated
                       else width / 2.0)
        if self.calibrated:
            bearing = math.degrees(math.atan2(centre[0] - principal_x,
                                              float(self.camera_matrix[0, 0])))
        else:
            bearing = ((centre[0] - principal_x) / (width / 2.0)
                       * (self.nominal_hfov_deg / 2.0))

        # A square marker viewed square-on has four equal sides. The spread of
        # side lengths is therefore a direct, calibration-free read on how
        # oblique the view is, and oblique views give the least stable poses.
        sides = [float(np.linalg.norm(corners[i] - corners[(i + 1) % 4]))
                 for i in range(4)]
        squareness = min(sides) / max(max(sides), 1e-6)

        range_m, yaw_deg, residual = None, None, None
        if self.calibrated:
            range_m, yaw_deg, residual = self._solve_pose(corners)

        confidence = self._score(area_fraction, squareness, residual)
        return MarkerSighting(marker_id=marker_id, centre_px=centre,
                              area_px=area, bearing_deg=bearing,
                              confidence=confidence, range_m=range_m,
                              yaw_deg=yaw_deg, calibrated=self.calibrated,
                              corners=corners)

    def _solve_pose(self, corners: np.ndarray):
        """Range and yaw by planar-square PnP, plus the fit's own residual."""
        half = self.marker_length_m / 2.0
        object_points = np.array([[-half, half, 0], [half, half, 0],
                                  [half, -half, 0], [-half, -half, 0]],
                                 dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            object_points, corners.astype(np.float64), self.camera_matrix,
            self.distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None, None, None

        projected, _ = cv2.projectPoints(object_points, rvec, tvec,
                                         self.camera_matrix, self.distortion)
        residual = float(np.linalg.norm(
            projected.reshape(4, 2) - corners, axis=1).mean())
        rotation, _ = cv2.Rodrigues(rvec)
        # Rotation about the camera's vertical axis: how far the marker's face
        # is turned away from us, which is what decides the approach arc.
        yaw = math.degrees(math.atan2(-rotation[2, 0],
                                      math.hypot(rotation[0, 0], rotation[1, 0])))
        return float(np.linalg.norm(tvec)), yaw, residual

    def _score(self, area_fraction: float, squareness: float,
               residual: float | None) -> float:
        """
        Combine the ways a sighting can be untrustworthy.

        Each term is a multiplier in 0-1, so any single bad signal drags the
        result down rather than being averaged away by two good ones.
        """
        if area_fraction < MIN_AREA_FRACTION:
            return 0.0
        # Saturates: past a few percent of frame, more size stops helping.
        size_term = min(1.0, math.sqrt(area_fraction / 0.01))
        # Below ~0.5 the marker is edge-on enough that corners smear together.
        shape_term = max(0.0, (squareness - 0.35) / 0.45)
        score = size_term * min(1.0, shape_term)

        if not self.calibrated:
            # Scale, do not clamp. min(score, ceiling) made every uncalibrated
            # sighting read exactly the ceiling -- a 40px marker scored the
            # same as a 260px one -- which leaves the behaviour tree's abort
            # and retry thresholds nothing to discriminate on.
            return float(score * UNCALIBRATED_CEILING)
        if residual is not None:
            # 2px of residual on a sighting is already suspect.
            score *= max(0.1, 1.0 - residual / 2.0)
        if self.reprojection_error:
            # The calibration's own error bounds how good any pose from it is.
            score *= max(0.3, 1.0 - self.reprojection_error / 2.0)
        return float(max(0.0, min(1.0, score)))

    def dock(self, frame: np.ndarray, target_id: int | None = None,
             stop_distance_m: float = 0.25) -> DockingCommand:
        """
        Turn-and-advance guidance toward a marker.

        Args:
            frame: Camera image.
            target_id: Marker to dock with; None takes the most confident.
            stop_distance_m: How far short of the marker to stop.
        """
        sightings = self.detect(frame)
        if not sightings:
            return DockingCommand(reason="no marker detected")

        if target_id is not None:
            matching = [s for s in sightings if s.marker_id == target_id]
            if not matching:
                seen = sorted({s.marker_id for s in sightings})
                return DockingCommand(
                    reason=f"marker {target_id} not visible (saw {seen})")
            sighting = matching[0]
        else:
            sighting = sightings[0]

        warnings = []
        if not self.calibrated:
            warnings.append("uncalibrated: bearing is approximate, no range")
        if sighting.range_m is not None and sighting.range_m < stop_distance_m:
            return DockingCommand(turn_deg=0.0, advance_m=0.0,
                                  confidence=sighting.confidence,
                                  reason="within stop distance",
                                  sighting=sighting, warnings=warnings)

        advance = (max(0.0, sighting.range_m - stop_distance_m)
                   if sighting.range_m is not None else None)
        if advance is None:
            warnings.append("no advance distance without calibration")
        if sighting.confidence < 0.25:
            warnings.append("low confidence; expect the tree to retry")

        return DockingCommand(turn_deg=sighting.bearing_deg, advance_m=advance,
                              confidence=sighting.confidence,
                              reason=f"marker {sighting.marker_id}",
                              sighting=sighting, warnings=warnings)

    def draw(self, frame: np.ndarray,
             sightings: list[MarkerSighting]) -> np.ndarray:
        """Overlay sightings for the dashboard."""
        canvas = frame.copy() if frame.ndim == 3 else cv2.cvtColor(
            frame, cv2.COLOR_GRAY2BGR)
        for sighting in sightings:
            if sighting.corners is None:
                continue
            points = sighting.corners.astype(np.int32).reshape(-1, 1, 2)
            colour = (0, 200, 0) if sighting.confidence >= 0.5 else (0, 165, 255)
            cv2.polylines(canvas, [points], True, colour, 2)
            label = f"id{sighting.marker_id} {sighting.confidence:.2f}"
            if sighting.range_m is not None:
                label += f" {sighting.range_m:.2f}m"
            label += f" {sighting.bearing_deg:+.1f}deg"
            cv2.putText(canvas, label,
                        (int(sighting.centre_px[0]) - 60,
                         int(sighting.centre_px[1]) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        return canvas
