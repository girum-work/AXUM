"""
docking.py — closed-loop visual docking onto an ArUco marker.

WHY A STATE MACHINE AND NOT A CONTROLLER LOOP: the rover's motion interface is
open-loop speed only -- drive/forward/turn_left/turn_right/stop, with no "turn
30 degrees" or "advance 200mm". So docking cannot be planned, only servoed:
issue a short pulse, look again, decide again. That also means the decision
logic can be a pure function of what the camera sees, which is the only way to
test it without a rover on a bench.

`step()` takes a frame and returns one motion pulse. It touches no hardware.

SAFETY PROPERTIES, each of which has a test:
    refuses to advance uncalibrated   bearing survives a guessed field of view;
                                      range does not. Turning to face a marker
                                      on an approximate bearing is recoverable,
                                      driving an unknown distance at it is not.
    stops on the proximity sensor     below the supervisor's collision
                                      distance, forward is never issued
    gives up rather than wandering    consecutive misses and a total step
                                      budget both terminate the attempt
    ignores low-confidence sightings  a marker seen badly is treated as not
                                      seen, not as a target
    every pulse is bounded            no command outlives one observation

The caller must set `drive_active` on the blackboard around these pulses. The
supervisor's front-collision and wheel-stall guards only run while that flag is
true, so a caller that forgets it disables both -- see the note in
mission_tree._action_navigate, which is where that requirement is written down.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from loguru import logger

from src.mission.marker_nav import DockingCommand, MarkerNavigator


class DockState(str, Enum):
    SEARCH = "search"
    ALIGN = "align"
    APPROACH = "approach"
    DOCKED = "docked"
    FAILED = "failed"


@dataclass
class DockingStep:
    """One pulse. `action` names a DriveController method, or "stop"."""

    action: str
    speed: int
    duration_s: float
    state: DockState
    confidence: float
    reason: str
    range_m: float | None = None
    bearing_deg: float | None = None

    @property
    def finished(self) -> bool:
        return self.state in (DockState.DOCKED, DockState.FAILED)

    @property
    def succeeded(self) -> bool:
        return self.state is DockState.DOCKED

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "speed": self.speed,
                "duration_s": round(self.duration_s, 3),
                "state": self.state.value,
                "confidence": round(self.confidence, 3),
                "reason": self.reason,
                "range_m": None if self.range_m is None else round(self.range_m, 3),
                "bearing_deg": (None if self.bearing_deg is None
                                else round(self.bearing_deg, 2))}


class DockingController:
    """
    Args:
        navigator: Supplies sightings. Its `calibrated` flag decides whether
            approach is permitted at all.
        target_id: Marker to dock with. Never substituted if absent.
        stop_distance_m: Range at which to declare success.
        align_tolerance_deg: Bearing within which approach may begin.
        min_confidence: Below this a sighting counts as a miss.
        max_misses: Consecutive misses before giving up.
        max_steps: Total pulses before giving up.
        collision_distance_cm: Forward is refused below this proximity reading.
    """

    def __init__(self, navigator: MarkerNavigator, target_id: int | None = None,
                 stop_distance_m: float = 0.25,
                 align_tolerance_deg: float = 6.0,
                 min_confidence: float = 0.20,
                 max_misses: int = 8, max_steps: int = 120,
                 collision_distance_cm: float = 8.0,
                 turn_speed: int = 110, drive_speed: int = 130) -> None:
        self.navigator = navigator
        self.target_id = target_id
        self.stop_distance_m = stop_distance_m
        self.align_tolerance_deg = align_tolerance_deg
        self.min_confidence = min_confidence
        self.max_misses = max_misses
        self.max_steps = max_steps
        self.collision_distance_cm = collision_distance_cm
        self.turn_speed = turn_speed
        self.drive_speed = drive_speed
        self.reset()

    def reset(self) -> None:
        self.state = DockState.SEARCH
        self.steps = 0
        self.misses = 0
        self.last_command: DockingCommand | None = None

    def _halt(self, state: DockState, reason: str,
              confidence: float = 0.0) -> DockingStep:
        self.state = state
        return DockingStep("stop", 0, 0.0, state, confidence, reason)

    def step(self, frame: np.ndarray,
             front_distance_cm: float | None = None) -> DockingStep:
        """Decide the next pulse from one frame."""
        if self.state in (DockState.DOCKED, DockState.FAILED):
            return self._halt(self.state, "already finished")

        self.steps += 1
        if self.steps > self.max_steps:
            return self._halt(DockState.FAILED,
                              f"step budget exhausted ({self.max_steps})")

        command = self.navigator.dock(frame, self.target_id,
                                      self.stop_distance_m)
        self.last_command = command
        sighting = command.sighting

        # A marker seen badly is not a target. Treating a 0.05-confidence
        # sighting as real is how a rover drives at a reflection.
        if sighting is None or command.confidence < self.min_confidence:
            self.misses += 1
            if self.misses > self.max_misses:
                return self._halt(
                    DockState.FAILED,
                    f"lost marker for {self.misses} frames ({command.reason})")
            self.state = DockState.SEARCH
            # Sweep in the last known direction so a marker that drifted off
            # one edge is not hunted towards the other.
            towards = getattr(self, "_last_bearing", 0.0)
            action = "turn_right" if towards >= 0 else "turn_left"
            return DockingStep(action, self.turn_speed, 0.20, DockState.SEARCH,
                               command.confidence,
                               f"searching ({command.reason})")

        self.misses = 0
        self._last_bearing = sighting.bearing_deg

        if front_distance_cm is not None and front_distance_cm < self.collision_distance_cm:
            return self._halt(DockState.FAILED,
                              f"obstacle at {front_distance_cm:.1f}cm",
                              command.confidence)

        if abs(sighting.bearing_deg) > self.align_tolerance_deg:
            self.state = DockState.ALIGN
            action = "turn_right" if sighting.bearing_deg > 0 else "turn_left"
            # Pulse proportional to the error, bounded so a large bearing
            # cannot produce a long blind turn.
            duration = min(0.35, 0.04 + abs(sighting.bearing_deg) / 90.0 * 0.30)
            return DockingStep(action, self.turn_speed, duration,
                               DockState.ALIGN, command.confidence,
                               f"aligning {sighting.bearing_deg:+.1f}deg",
                               sighting.range_m, sighting.bearing_deg)

        if sighting.range_m is None:
            # Aligned, and that is as far as an uncalibrated camera can take
            # us. Reported as a failure with a reason, not as a near-success.
            return self._halt(
                DockState.FAILED,
                "aligned, but range needs calibration "
                "(run scripts/calibrate_camera.py)", command.confidence)

        if sighting.range_m <= self.stop_distance_m:
            self.state = DockState.DOCKED
            return DockingStep("stop", 0, 0.0, DockState.DOCKED,
                               command.confidence,
                               f"docked at {sighting.range_m:.2f}m",
                               sighting.range_m, sighting.bearing_deg)

        self.state = DockState.APPROACH
        remaining = sighting.range_m - self.stop_distance_m
        # Shorter pulses as the marker nears, so the last approach cannot
        # overshoot past the stop distance between two observations.
        duration = float(min(0.40, max(0.10, remaining * 0.5)))
        return DockingStep("forward", self.drive_speed, duration,
                           DockState.APPROACH, command.confidence,
                           f"approaching, {remaining:.2f}m to go",
                           sighting.range_m, sighting.bearing_deg)

    def run(self, frame_source, hardware=None,
            front_distance: Any = None) -> DockingStep:
        """
        Servo until docked or failed.

        Args:
            frame_source: Callable returning the current frame, or None.
            hardware: Object exposing forward/turn_left/turn_right/stop. If
                None the loop decides without moving, which is how it is
                exercised in tests.
            front_distance: Callable returning the proximity reading in cm.
        """
        result = self._halt(DockState.FAILED, "no frames")
        while True:
            frame = frame_source()
            if frame is None:
                result = self._halt(DockState.FAILED, "camera returned nothing")
                break
            distance = front_distance() if callable(front_distance) else front_distance
            result = self.step(frame, distance)

            if hardware is not None:
                self._actuate(hardware, result)
            if result.finished:
                break

        if hardware is not None:
            hardware.stop()
        logger.info(f"Docking {result.state.value}: {result.reason}")
        return result

    @staticmethod
    def _actuate(hardware, step: DockingStep) -> None:
        """Issue one bounded pulse, then stop."""
        import time

        if step.action == "stop" or step.duration_s <= 0:
            hardware.stop()
            return
        getattr(hardware, step.action)(step.speed)
        time.sleep(step.duration_s)
        hardware.stop()
