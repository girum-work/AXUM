"""
Tests for the OpenCV crack detector.

The detector had no tests, and the gap hid a real fault: severity saturated at
1.000 for every real image, so is_treatable was constant False and the
conservation output carried no information. These tests pin the properties that
failure violated.

Deliberately not asserted: absolute crack counts on real photographs. The
detector currently returns 488-1410 contours on natural stone, which is texture
rather than damage. Fixing that needs labelled ground truth, and freezing the
present numbers would only make the bug permanent.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.crack_detection.detector import CrackDetector


def blank_stone(width: int = 400, height: int = 300, value: int = 160) -> np.ndarray:
    """Uniform grey plate with no cracks."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def with_cracks(count: int, width: int = 400, height: int = 300) -> np.ndarray:
    """Blank plate with `count` dark lines drawn across it."""
    image = blank_stone(width, height)
    for index in range(count):
        x = int(width * (index + 1) / (count + 1))
        cv2.line(image, (x, 20), (x + 15, height - 20), (40, 40, 40), 2)
    return image


@pytest.fixture(scope="module")
def detector() -> CrackDetector:
    return CrackDetector()


def test_severity_is_bounded(detector):
    """Score must stay in [0, 1] whatever the input."""
    for image in (blank_stone(), with_cracks(1), with_cracks(12)):
        score = detector.detect(image).severity_score
        assert 0.0 <= score <= 1.0


def test_clean_surface_scores_low(detector):
    """An undamaged plate must not be reported as damaged."""
    result = detector.detect(blank_stone())
    assert result.severity_score < 0.1
    assert result.crack_count == 0


def test_severity_increases_with_damage(detector):
    """More cracks must not score lower than fewer."""
    scores = [detector.detect(with_cracks(n)).severity_score for n in (1, 4, 10)]
    assert scores == sorted(scores)
    assert scores[-1] > scores[0]


def test_severity_discriminates(detector):
    """
    Guards the original bug: the score must not collapse to one value.

    Previously length was normalised by the image diagonal, which assumed a
    single crack spanning the frame; hundreds of contours drove every real
    image to exactly 1.000.
    """
    scores = {round(detector.detect(with_cracks(n)).severity_score, 3)
              for n in (1, 3, 6, 10)}
    assert len(scores) > 1, "severity is constant across damage levels"
    assert scores != {1.0}


def test_treatable_flag_can_be_true(detector):
    """
    is_treatable must be reachable, or the conservation decision is inert.

    This is the check that would have caught the saturation bug: every test
    image reported not-treatable regardless of condition.
    """
    payload = detector.detect(with_cracks(1)).to_dict()
    assert payload["is_treatable"] is True


def test_detection_is_deterministic(detector):
    """Same input, same result -- no RNG in the OpenCV path."""
    image = with_cracks(5)
    first, second = detector.detect(image), detector.detect(image)
    assert first.crack_count == second.crack_count
    assert first.severity_score == pytest.approx(second.severity_score)


def test_overlay_matches_input_size(detector):
    """Overlay is drawn for the dashboard, so its geometry must line up."""
    image = with_cracks(3)
    result = detector.detect(image)
    assert result.overlay.shape == image.shape
    assert result.mask.shape[:2] == image.shape[:2]


def test_severity_is_scale_aware(detector):
    """
    Document how much resolution shifts the score.

    MIN_CRACK_AREA and MIN_CRACK_LEN are pixel thresholds, so the same surface
    photographed larger clears them more easily. The bound is loose because it
    records current behaviour rather than endorsing it: tightening it needs the
    thresholds expressed relative to image size.
    """
    small = with_cracks(4, width=400, height=300)
    large = cv2.resize(small, (800, 600), interpolation=cv2.INTER_LINEAR)
    difference = abs(detector.detect(small).severity_score
                     - detector.detect(large).severity_score)
    assert difference < 0.5, f"severity moved {difference:.3f} on a pure rescale"
