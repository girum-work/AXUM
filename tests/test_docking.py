"""
Tests for closed-loop ArUco docking.

Every safety property in docking.py's docstring is asserted here, because this
is the one module in the project that can drive a physical rover at something.
The decision logic is a pure function of the frame, which is what makes that
testable at all -- no hardware is touched by any of these.

Deliberately not asserted: how many pulses a real approach takes. That depends
on wheel slip, surface and battery, none of which a unit test can see.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.mission.docking import DockingController, DockState
from src.mission.marker_nav import MarkerNavigator


def scene(marker_id: int = 0, size: int = 160, centre_x: int = 320):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    frame = np.full((480, 640), 255, np.uint8)
    x0, y0 = centre_x - size // 2, 240 - size // 2
    frame[y0:y0 + size, x0:x0 + size] = marker
    return frame


def blank():
    return np.full((480, 640), 255, np.uint8)


class FakeRange(MarkerNavigator):
    """
    A calibrated navigator, without needing a calibrated camera.

    Range is the thing under test in APPROACH, so it is injected rather than
    solved. Faking the geometry instead would test cv2, not this state machine.
    """

    def __init__(self, range_m: float, **kwargs):
        super().__init__(**kwargs)
        self._range_m = range_m
        self.camera_matrix = np.array([[900.0, 0, 320], [0, 900.0, 240],
                                       [0, 0, 1]])
        self.distortion = np.zeros((1, 5))
        self.reprojection_error = 0.2

    def _solve_pose(self, corners):
        return self._range_m, 0.0, 0.15


@pytest.fixture
def uncalibrated():
    return DockingController(MarkerNavigator(), target_id=0)


def test_refuses_to_advance_without_calibration(uncalibrated):
    """The property that matters most: never drive an unknown distance."""
    for _ in range(6):
        step = uncalibrated.step(scene())
        assert step.action != "forward", "advanced on a guessed range"
        if step.finished:
            break
    assert step.state is DockState.FAILED
    assert "calibrat" in step.reason


def test_aligns_before_it_gives_up_uncalibrated(uncalibrated):
    """Turning on an approximate bearing is safe, so it should still do it."""
    step = uncalibrated.step(scene(centre_x=170))
    assert step.state is DockState.ALIGN
    assert step.action == "turn_left"


def test_turn_direction_follows_bearing_sign(uncalibrated):
    assert uncalibrated.step(scene(centre_x=170)).action == "turn_left"
    uncalibrated.reset()
    assert uncalibrated.step(scene(centre_x=470)).action == "turn_right"


def test_approaches_and_docks_when_calibrated():
    far = DockingController(FakeRange(2.0), target_id=0, stop_distance_m=0.25)
    step = far.step(scene())
    assert step.state is DockState.APPROACH and step.action == "forward"

    near = DockingController(FakeRange(0.2), target_id=0, stop_distance_m=0.25)
    step = near.step(scene())
    assert step.state is DockState.DOCKED and step.succeeded
    assert step.action == "stop"


def test_pulses_shorten_on_the_final_approach():
    """A long pulse near the marker could overshoot between observations."""
    far = DockingController(FakeRange(3.0), target_id=0).step(scene())
    close = DockingController(FakeRange(0.4), target_id=0).step(scene())
    assert close.duration_s < far.duration_s
    assert far.duration_s <= 0.40


def test_proximity_sensor_overrides_the_camera():
    controller = DockingController(FakeRange(2.0), target_id=0,
                                   collision_distance_cm=8.0)
    step = controller.step(scene(), front_distance_cm=5.0)
    assert step.state is DockState.FAILED and step.action == "stop"
    assert "obstacle" in step.reason


def test_gives_up_rather_than_wandering():
    controller = DockingController(MarkerNavigator(), target_id=0,
                                   max_misses=3)
    for _ in range(3):
        assert controller.step(blank()).state is DockState.SEARCH
    assert controller.step(blank()).state is DockState.FAILED


def test_step_budget_terminates_a_stuck_attempt():
    controller = DockingController(FakeRange(50.0), target_id=0, max_steps=5)
    for _ in range(5):
        controller.step(scene())
    exhausted = controller.step(scene())
    assert exhausted.state is DockState.FAILED
    assert "budget" in exhausted.reason
    assert controller.step(scene()).reason == "already finished"


def test_low_confidence_sighting_counts_as_a_miss():
    """A marker seen badly is not a target to drive at."""
    controller = DockingController(FakeRange(2.0), target_id=0,
                                   min_confidence=0.99)
    assert controller.step(scene()).state is DockState.SEARCH


def test_wrong_marker_is_never_substituted():
    controller = DockingController(FakeRange(2.0), target_id=3)
    assert controller.step(scene(marker_id=0)).state is DockState.SEARCH


def test_run_stops_the_hardware_even_when_it_fails():
    class Recorder:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            def record(*args):
                self.calls.append(name)
            return record

    hardware = Recorder()
    controller = DockingController(MarkerNavigator(), target_id=0, max_misses=1)
    result = controller.run(frame_source=blank, hardware=hardware)
    assert result.state is DockState.FAILED
    assert hardware.calls[-1] == "stop", "drive left running after a failure"


def test_finished_controller_stays_finished():
    controller = DockingController(FakeRange(0.1), target_id=0)
    assert controller.step(scene()).succeeded
    after = controller.step(scene(centre_x=100))
    assert after.action == "stop" and after.state is DockState.DOCKED
