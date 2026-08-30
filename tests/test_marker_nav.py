"""
Tests for ArUco marker navigation.

These pin the properties the behaviour tree depends on. behavior_tree.py
branches on `navigate_confidence` -- hard failure below `abort_below`, retry
between that and `retry_below` -- so a confidence that does not discriminate is
worse than no confidence at all. One defect was already found this way: capping
the uncalibrated score with min() made every sighting read exactly 0.400,
whatever its size.

Not asserted: absolute range accuracy. That needs a calibrated camera and a
physically measured marker, and freezing numbers from placeholder intrinsics
would repeat the mistake pi_aruco_benchmark.py documents in its own comments.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.mission.marker_nav import UNCALIBRATED_CEILING, MarkerNavigator


def scene(marker_id: int = 0, size: int = 160, centre_x: int = 320,
          centre_y: int = 240, width: int = 640, height: int = 480):
    """A marker pasted onto white, with a quiet zone around it."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    frame = np.full((height, width), 255, np.uint8)
    x0, y0 = centre_x - size // 2, centre_y - size // 2
    frame[y0:y0 + size, x0:x0 + size] = marker
    return frame


@pytest.fixture(scope="module")
def navigator() -> MarkerNavigator:
    return MarkerNavigator()


def test_finds_marker_and_reads_its_id(navigator):
    for marker_id in (0, 1, 7):
        found = navigator.detect(scene(marker_id))
        assert [s.marker_id for s in found] == [marker_id]


def test_blank_frame_reports_nothing(navigator):
    assert navigator.detect(np.full((480, 640), 255, np.uint8)) == []
    assert navigator.dock(np.full((480, 640), 255, np.uint8)).confidence == 0.0


def test_bearing_sign_follows_marker_across_frame(navigator):
    left = navigator.dock(scene(centre_x=160)).turn_deg
    centre = navigator.dock(scene(centre_x=320)).turn_deg
    right = navigator.dock(scene(centre_x=480)).turn_deg
    assert left < centre < right, "bearing must increase left to right"
    assert left < 0 < right, "sign must flip about the optical axis"
    assert abs(centre) < 1.0, "a centred marker must read near zero"


def test_confidence_discriminates_by_size(navigator):
    """The defect this caught: a clamp made every sighting read the ceiling."""
    small = navigator.detect(scene(size=36))[0].confidence
    large = navigator.detect(scene(size=200))[0].confidence
    assert small < large, "a distant marker must not score like a near one"
    assert 0.0 < small < 1.0


def test_uncalibrated_confidence_stays_under_the_ceiling(navigator):
    """Without intrinsics the bearing rests on an assumed FOV, so cap belief."""
    for size in (60, 120, 240, 320):
        found = navigator.detect(scene(size=size))
        if found:
            assert found[0].confidence <= UNCALIBRATED_CEILING + 1e-6


def test_uncalibrated_offers_no_range(navigator):
    assert navigator.calibrated is False
    sighting = navigator.detect(scene())[0]
    assert sighting.range_m is None and sighting.yaw_deg is None
    command = navigator.dock(scene())
    assert command.advance_m is None
    assert any("calibrat" in w for w in command.warnings)


def test_missing_target_is_reported_not_substituted(navigator):
    """Docking with the wrong marker would drive at the wrong object."""
    command = navigator.dock(scene(marker_id=0), target_id=3)
    assert command.confidence == 0.0
    assert "not visible" in command.reason
    assert command.sighting is None


def test_detection_is_deterministic(navigator):
    frame = scene()
    first, second = navigator.dock(frame), navigator.dock(frame)
    assert first.turn_deg == pytest.approx(second.turn_deg)
    assert first.confidence == pytest.approx(second.confidence)


def test_overlay_matches_frame_geometry(navigator):
    frame = scene()
    canvas = navigator.draw(frame, navigator.detect(frame))
    assert canvas.shape[:2] == frame.shape[:2]
    assert canvas.ndim == 3, "dashboard overlay must be colour"


def test_payload_is_json_friendly(navigator):
    payload = navigator.dock(scene()).to_dict()
    assert set(payload) >= {"turn_deg", "advance_m", "confidence", "reason"}
    assert isinstance(payload["confidence"], float)
    assert payload["sighting"]["marker_id"] == 0
