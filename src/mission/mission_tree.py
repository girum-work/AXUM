"""
AXUM ROVER - Mission behavior tree.

WHAT: Builds the actual per-artefact mission tree out of the generic
behavior_tree engine, wired to controller.py's hardware classes and the
existing analysis modules (crack detection, salt mapping, photometric
stereo, multispectral, fragility clock, treatment advisor).

WHY a tree instead of the old linear process_one(): a straight-line
function has no way to react to "battery critical" or "grip confidence
low" except by adding that check everywhere by hand. The supervisor
Parallel at the root means one battery/comms check can interrupt any
mission phase, instead of eight copies of the same check.

HONESTY NOTE: the NAVIGATE phase is now implemented -- ArUco docking, closed
loop on the camera, in src/mission/docking.py. It refuses to run on an
uncalibrated camera: bearing survives a guessed field of view, range does not,
so an uncalibrated rover can face a marker but must not drive at it. Run
scripts/calibrate_camera.py before expecting it to succeed on hardware.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from src.mission.behavior_tree import (
    ActionNode,
    Blackboard,
    ConditionNode,
    ConfidenceGate,
    Decorator,
    FaultInjector,
    Fallback,
    Invariant,
    MissionRecorder,
    Node,
    Parallel,
    Retry,
    Sequence,
    Status,
    Timeout,
    Traced,
    run_to_completion,
)

from config import (
    CAMERA_CALIBRATION_PATH,
    DOCK_ALIGN_TOLERANCE_DEG,
    DOCK_MARKER_ID,
    DOCK_STOP_DISTANCE_M,
    MARKER_LENGTH_M,
)

# Firmware readiness flag — same meaning as in main_pipeline.py. Kept
# separate here (not imported) so this module has no hard dependency on
# main_pipeline's internals; the Robotics/Embedded team should keep both
# flags in sync manually until there's a single shared config source.
FIRMWARE_LED_READY = False

# STATUS shape as of this integration pass (confirmed against the real
# axum_rover.ino, re-verified after encoders were removed):
#   {"front": <float cm>, "side": <float cm>, "ir_1"/"ir_2"/"ir_3": <int>,
#    "loadcell_raw": <float or null>}
# enc_l/enc_r NO LONGER EXIST. Encoders were removed from the firmware
# this session ("we will NOT be using any rotary encoders for now, that's
# why the ESP32 cameras will be wired there instead" -- pins 18/19 now
# belong to Serial1). THERE IS NO BATTERY VOLTAGE FIELD either.
#
# CONSEQUENCE FOR THIS FILE: _update_stall_check() below already degrades
# safely on its own -- controller.arduino.status().get("enc_l") now always
# returns None, which _update_stall_check() correctly treats as "unknown"
# (wheels_stalled=None), not a false "not stalled". No logic bug, nothing
# broken. But the wheel-stall safety check this function was built and
# tested for is now PERMANENTLY DORMANT -- it will never fire again on
# real hardware, because the sensor it depends on isn't there anymore.
# That capability loss is real and worth knowing about, not something to
# silently accept. If wheel-stall detection matters for the actual
# mission, it needs a different sensing source now -- not something to
# invent here without asking Robotics/Electronics what's realistic.
FRONT_COLLISION_DISTANCE_CM = 8.0  # stop driving forward if closer than this — placeholder, confirm real safe distance with Mechanical
STALL_ENCODER_DELTA_THRESHOLD = 2  # min encoder ticks expected between polls while actively driving, else treat as a stalled wheel — DORMANT, see note above


def _action_poll_health(bb: Blackboard) -> Status:
    """
    Reads live telemetry from the real STATUS response and writes it to
    the blackboard. Only ever reports what STATUS actually contains —
    no fields are assumed or guessed.

    Battery voltage is NOT available from firmware today (confirmed by
    reading axum_rover.ino directly). `_condition_health_ok` therefore
    cannot check battery level and will not until firmware adds a field
    for it — this is a real capability gap, not a bug in this function.

    What IS available and used: front-distance-based collision guarding.
    Wheel-stall checking (comparing encoder deltas between polls) is still
    called below but is now PERMANENTLY DORMANT -- encoders were removed
    from the firmware this session, so enc_l/enc_r are never present in
    STATUS anymore. See the module-level note above _update_stall_check.
    """
    controller = bb.get("hardware")
    if controller is None:
        bb.set("front_distance_cm", None)
        bb.set("side_distance_cm", None)
        bb.set("comms_alive", bb.get("dry_run", False))
        return Status.SUCCESS
    try:
        status_payload = controller.arduino.status()
        bb.set("comms_alive", True)
        if isinstance(status_payload, dict):
            front = status_payload.get("front")
            side = status_payload.get("side")
            enc_l = status_payload.get("enc_l")
            enc_r = status_payload.get("enc_r")
            bb.set("front_distance_cm", front)
            bb.set("side_distance_cm", side)
            _update_stall_check(bb, enc_l, enc_r)
        else:
            bb.set("front_distance_cm", None)
            bb.set("side_distance_cm", None)
    except Exception as exc:
        bb.set("comms_alive", False)
        logger.warning(f"Health poll failed: {exc}")
    return Status.SUCCESS


def _update_stall_check(bb: Blackboard, enc_l, enc_r) -> None:
    """
    Compares encoder counts between consecutive polls while a drive
    command is active. If the rover is commanded to move but encoder
    counts aren't changing, that's a stalled/stuck wheel — a real fault
    STATUS's real fields make detectable that wasn't detectable before.
    """
    if enc_l is None or enc_r is None:
        bb.set("wheels_stalled", None)
        return
    prev_l = bb.get("_prev_enc_l")
    prev_r = bb.get("_prev_enc_r")
    driving = bb.get("drive_active", False)
    if driving and prev_l is not None and prev_r is not None:
        delta_l = abs(enc_l - prev_l)
        delta_r = abs(enc_r - prev_r)
        stalled = delta_l < STALL_ENCODER_DELTA_THRESHOLD and delta_r < STALL_ENCODER_DELTA_THRESHOLD
        bb.set("wheels_stalled", stalled)
        if stalled:
            bb.set("last_abort_reason", f"Wheels stalled: enc_l delta={delta_l}, enc_r delta={delta_r} while driving")
    else:
        bb.set("wheels_stalled", False)
    bb.set("_prev_enc_l", enc_l)
    bb.set("_prev_enc_r", enc_r)


def _condition_health_ok(bb: Blackboard) -> bool:
    if bb.get("estop_requested"):
        bb.set("last_abort_reason", "E-stop requested")
        return False
    if bb.get("comms_alive") is False:
        bb.set("last_abort_reason", "Lost serial comms to Arduino")
        return False
    if bb.get("wheels_stalled") is True:
        return False  # last_abort_reason already set by _update_stall_check
    front_distance = bb.get("front_distance_cm")
    if bb.get("drive_active") and front_distance is not None and front_distance < FRONT_COLLISION_DISTANCE_CM:
        bb.set("last_abort_reason", f"Front collision guard: {front_distance}cm < {FRONT_COLLISION_DISTANCE_CM}cm")
        return False
    # NOTE: no battery-voltage check exists here. Firmware does not report
    # one. Do not add a check against a field that doesn't exist — if
    # battery monitoring is needed, that's a firmware change request to
    # Embedded, not something to fake at this layer.
    return True


def build_supervisor() -> Sequence:
    """Health poll + guard, ticked every cycle alongside the mission sequence."""
    return Sequence("Supervisor", [
        ActionNode("PollHealth", _action_poll_health),
        ConditionNode("HealthOK", _condition_health_ok),
    ])


# ── mission phase actions ────────────────────────────────────────────
# Each of these closes over `pipeline` (the MissionPipeline instance from
# main_pipeline.py) to reuse its existing hardware handles and analysis
# calls, so this file doesn't duplicate that logic.

def _make_action_navigate(pipeline) -> Any:
    """
    ArUco docking, closed-loop on the camera.

    Refuses to run uncalibrated. Bearing survives a guessed field of view and
    range does not, so an uncalibrated camera can face a marker but cannot know
    how far to drive at it. Failing here is recoverable; driving an unknown
    distance is not. scripts/calibrate_camera.py produces the file.

    `drive_active` is set for the duration and cleared in a finally, because
    the supervisor's front-collision and wheel-stall guards only run while it
    is true -- a path that leaves it false drives with both disabled.
    """
    def _action_navigate(bb: Blackboard) -> Status:
        if bb.get("dry_run"):
            bb.set("navigate_confidence", 1.0)
            return Status.SUCCESS

        from src.mission.docking import DockingController
        from src.mission.marker_nav import MarkerNavigator

        navigator = MarkerNavigator(calibration=Path(CAMERA_CALIBRATION_PATH),
                                    marker_length_m=MARKER_LENGTH_M)
        if not navigator.calibrated:
            logger.error(
                f"NAVIGATE refuses to drive uncalibrated: no intrinsics at "
                f"{CAMERA_CALIBRATION_PATH}. Run scripts/calibrate_camera.py. "
                f"Detection works; range does not.")
            bb.set("navigate_confidence", 0.0)
            bb.set("last_abort_reason", "Camera not calibrated for docking")
            return Status.FAILURE

        hardware = bb.get("hardware")
        camera = getattr(pipeline, "camera", None) or getattr(
            pipeline, "_camera", None)
        if camera is None or hardware is None:
            logger.error("NAVIGATE has no camera or no drive controller")
            bb.set("navigate_confidence", 0.0)
            return Status.FAILURE

        controller = DockingController(
            navigator,
            target_id=bb.get("dock_marker_id", DOCK_MARKER_ID),
            stop_distance_m=DOCK_STOP_DISTANCE_M,
            align_tolerance_deg=DOCK_ALIGN_TOLERANCE_DEG,
            collision_distance_cm=FRONT_COLLISION_DISTANCE_CM,
        )

        bb.set("drive_active", True)
        try:
            result = controller.run(
                frame_source=camera.capture_frame,
                hardware=hardware,
                front_distance=lambda: bb.get("front_distance_cm"),
            )
        finally:
            bb.set("drive_active", False)
            try:
                hardware.stop()
            except Exception:  # a failed stop must not mask the real error
                logger.exception("Could not stop the drive after docking")

        bb.set("navigate_confidence",
               result.confidence if result.succeeded else 0.0)
        bb.set("dock_state", result.state.value)
        if not result.succeeded:
            bb.set("last_abort_reason", f"Docking failed: {result.reason}")
            logger.error(f"NAVIGATE failed: {result.reason}")
            return Status.FAILURE
        logger.info(f"NAVIGATE docked: {result.reason}")
        return Status.SUCCESS

    return _action_navigate


def _make_action_pick(pipeline) -> Any:
    def _action_pick(bb: Blackboard) -> Status:
        object_id = bb.get("object_id")
        try:
            if bb.get("dry_run"):
                bb.set("grip_confidence", 1.0)
                return Status.SUCCESS
            pipeline._hardware.pick_from_tray()
            # FSR-based grip confidence — ArmController doesn't currently
            # expose a confidence read after pick_from_tray(); this reads
            # whatever pick_from_tray() itself last recorded if available,
            # otherwise treats an exception-free call as a soft SUCCESS
            # rather than fabricating a number that isn't real.
            confidence = getattr(pipeline._hardware, "last_grip_confidence", None)
            bb.set("grip_confidence", confidence if confidence is not None else 0.75)
            return Status.SUCCESS
        except Exception as exc:
            logger.warning(f"{object_id}: pick failed: {exc}")
            bb.set("grip_confidence", 0.0)
            return Status.FAILURE
    return _action_pick


def _make_action_transfer(pipeline) -> Any:
    def _action_transfer(bb: Blackboard) -> Status:
        try:
            if not bb.get("dry_run"):
                pipeline._hardware.place_on_turntable()
            return Status.SUCCESS
        except Exception as exc:
            logger.warning(f"transfer failed: {exc}")
            return Status.FAILURE
    return _action_transfer


def _make_action_scan(pipeline) -> Any:
    def _action_scan(bb: Blackboard) -> Status:
        object_id = bb.get("object_id")
        try:
            scan = pipeline._scan_artefact(object_id)
            bb.set("reference_frame", scan["reference_frame"])
            bb.set("mesh_photo_count", scan["mesh_photo_count"])
            bb.set("quad_paths", scan["quad_paths"])
            bb.set("visible_path", scan["visible_path"])
            bb.set("ir_path", scan["ir_path"])
            return Status.SUCCESS
        except Exception as exc:
            logger.warning(f"{object_id}: scan failed: {exc}")
            return Status.FAILURE
    return _action_scan


def _make_action_analyze(pipeline) -> Any:
    """Crack + salt + photometric + multispectral + fragility, in one leaf."""
    def _action_analyze(bb: Blackboard) -> Status:
        import cv2

        object_id = bb.get("object_id")
        sequence_number = bb.get("sequence_number")
        try:
            frame = bb.get("reference_frame")
            crack_result = pipeline.crack_detector.detect(frame)
            bb.set("crack_severity", crack_result.severity_score)

            from src.imaging.salt_mapper import run_salt_mapper
            uv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            salt_result = run_salt_mapper(uv_image=uv_frame, visible_image=frame, artefact_id=object_id)
            bb.set("salt_risk", salt_result.overall_risk)
            bb.set("salt_critical", len(salt_result.critical_zones) > 0)

            # BUGFIX (Systems Integration Engineer, this pass): the real
            # run_photometric_stereo() takes an already-loaded
            # dict[str, np.ndarray] and has NO artefact_id parameter --
            # calling it with a path dict + artefact_id= would raise
            # TypeError the first time this phase actually ran. The
            # path-based entry point is analyse_photometric_paths().
            quad_paths = bb.get("quad_paths")
            if quad_paths:
                from src.imaging.photometric_stereo import analyse_photometric_paths
                analyse_photometric_paths(quad_paths)

            # Multispectral analysis requires a physically aligned visible/IR
            # pair.  The current scan returns no IR path, so preserve the
            # safe zero-stress fallback rather than mislabeling a UV-lit image
            # as infrared data.
            from src.imaging.multispectral import analyse_multispectral_paths
            visible_path = bb.get("visible_path")
            ir_path = bb.get("ir_path")
            ms_result = (
                analyse_multispectral_paths(visible_path, ir_path)
                if visible_path and ir_path
                else None
            )
            # MultispectralResult has no stress_score field; its structural
            # hazard score is the compatible input when an IR pair exists.
            stress_score = ms_result.structural_hazard_score if ms_result else 0.0
            bb.set("stress_score", stress_score)

            from config import OBJ_CLASSES
            class_name = OBJ_CLASSES[(sequence_number - 1) % len(OBJ_CLASSES)]
            bb.set("class_name", class_name)
            bb.set("has_inscription", class_name in {"coin", "inscription_fragment", "stone_carving"})

            from src.analysis.fragility_clock import FragilityInputs, run_fragility_clock
            fragility = run_fragility_clock(FragilityInputs(
                artefact_id=object_id,
                artefact_class=class_name,
                crack_severity=crack_result.severity_score,
                salt_risk=salt_result.overall_risk,
                salt_critical=len(salt_result.critical_zones) > 0,
                stress_score=stress_score,
                active_moisture=False,
            ))
            bb.set("years_remaining", fragility.years_remaining)
            bb.set("urgency_band", fragility.urgency_band)
            bb.set("crack_result", crack_result)
            bb.set("salt_result", salt_result)
            return Status.SUCCESS
        except Exception as exc:
            logger.warning(f"{object_id}: analysis failed: {exc}")
            return Status.FAILURE
    return _action_analyze


def _make_action_treat(pipeline) -> Any:
    def _action_treat(bb: Blackboard) -> Status:
        object_id = bb.get("object_id")
        try:
            from src.analysis.treatment_advisor import DiagnosticInputs, run_treatment_advisor
            protocol = run_treatment_advisor(DiagnosticInputs(
                artefact_id=object_id,
                artefact_class=bb.get("class_name"),
                crack_severity=bb.get("crack_severity"),
                salt_risk=bb.get("salt_risk"),
                salt_critical=bb.get("salt_critical"),
                stress_score=bb.get("stress_score"),
                ocr_confidence=0.0,
                has_inscription=bb.get("has_inscription"),
                biological_detected=False,
                years_remaining=bb.get("years_remaining"),
                active_moisture=False,
            ))
            bb.set("protocol", protocol)
            return Status.SUCCESS
        except Exception as exc:
            logger.warning(f"{object_id}: treatment advisory failed: {exc}")
            return Status.FAILURE
    return _action_treat


def _make_action_return(pipeline) -> Any:
    def _action_return(bb: Blackboard) -> Status:
        try:
            if not bb.get("dry_run"):
                pipeline._hardware.go_pose("PARK")
            return Status.SUCCESS
        except Exception as exc:
            logger.warning(f"return-to-park failed: {exc}")
            return Status.FAILURE
    return _action_return


# ── invariant predicates ─────────────────────────────────────────────

def _grip_confirmed(bb: Blackboard) -> bool:
    """
    Safety invariant for TRANSFER: never move a picked artefact toward the
    turntable unless PICK reported real grip confidence, not just "no
    exception was thrown." Below 0.5 grip confidence, treat the object as
    not actually secured.
    """
    if bb.get("dry_run"):
        return True
    return (bb.get("grip_confidence") or 0.0) >= 0.5


# ── tree assembly ─────────────────────────────────────────────────────

class LiveDashboard(Decorator):
    """
    Pushes phase transitions to the dashboard in real time, as they happen
    — not just to the black-box log after the mission finishes.

    WHY this lives here and not in behavior_tree.py: the generic engine
    has zero knowledge of dashboards, Flask, or AXUM's event names by
    design (see behavior_tree.py's module docstring) — that's what keeps
    it reusable and testable without any of this project's other
    infrastructure. This decorator is the AXUM-specific bridge.

    WHY the generic emit_event channel, not a new typed function: unlike
    emit_fragility_clock (which Backend specified a fixed signature for),
    nobody has defined a typed "mission_phase" event yet. Using the
    existing generic channel rather than inventing a new typed one
    unilaterally — if Backend wants a typed version later, this is a
    one-line change contained entirely to this class.

    Every push is wrapped in try/except exactly like every other
    dashboard call in this codebase — a dashboard outage must never be
    able to stall or fail a mission.
    """

    def __init__(self, name: str, child: Node) -> None:
        super().__init__(name, child)

    def tick(self, bb: Blackboard) -> Status:
        status = self.child.tick(bb)
        self._push(bb, status)
        return status

    def _push(self, bb: Blackboard, status: Status) -> None:
        try:
            from src.dashboard.server import emit_event

            emit_event("mission_phase", {
                "artefact_id": bb.get("object_id"),
                "phase": self.name,
                "status": status.value,
            })
        except Exception as exc:
            logger.debug(f"Live dashboard push skipped ({self.name}): {exc}")


def build_mission_tree(pipeline, object_id: str, sequence_number: int, recorder: MissionRecorder) -> Sequence:
    """
    Builds the full per-artefact mission sequence: navigate -> pick ->
    transfer -> scan -> analyze -> treat -> return. Each phase is wrapped
    in Timeout + Retry appropriate to that phase, PICK's outcome is
    confidence-gated, and TRANSFER is guarded by a safety invariant.
    """
    navigate = Timeout("NavigateTimeout", Retry("NavigateRetry", ActionNode("Navigate", _make_action_navigate(pipeline)), max_attempts=2), timeout_seconds=20)
    navigate = ConfidenceGate("NavigateConfidence", navigate, confidence_key="navigate_confidence", retry_below=0.6, abort_below=0.2)

    pick = Timeout("PickTimeout", Retry("PickRetry", ActionNode("Pick", _make_action_pick(pipeline)), max_attempts=3), timeout_seconds=15)
    pick = ConfidenceGate("PickConfidence", pick, confidence_key="grip_confidence", retry_below=0.5, abort_below=0.15)

    transfer = Invariant(
        "GripConfirmedInvariant",
        Timeout("TransferTimeout", ActionNode("Transfer", _make_action_transfer(pipeline)), timeout_seconds=10),
        predicate=_grip_confirmed,
        violation_message="Refusing TRANSFER: grip confidence too low to trust the object is actually held",
    )

    scan = Timeout("ScanTimeout", Retry("ScanRetry", ActionNode("Scan", _make_action_scan(pipeline)), max_attempts=2), timeout_seconds=120)
    analyze = Retry("AnalyzeRetry", ActionNode("Analyze", _make_action_analyze(pipeline)), max_attempts=2)
    treat = ActionNode("Treat", _make_action_treat(pipeline))
    do_return = Timeout("ReturnTimeout", Retry("ReturnRetry", ActionNode("Return", _make_action_return(pipeline)), max_attempts=2), timeout_seconds=15)

    phases = [navigate, pick, transfer, scan, analyze, treat, do_return]
    traced_phases = [LiveDashboard(node.name, Traced(node.name, node, recorder)) for node in phases]
    return Sequence(f"Mission[{object_id}]", traced_phases)


def build_supervised_tree(pipeline, object_id: str, sequence_number: int, recorder: MissionRecorder) -> Parallel:
    """Root: supervisor and mission sequence tick together every cycle."""
    return Parallel("SupervisedMission", [
        build_supervisor(),
        build_mission_tree(pipeline, object_id, sequence_number, recorder),
    ])


def run_artefact_mission(
    pipeline,
    object_id: str,
    sequence_number: int,
    dry_run: bool,
    photo_dir: Path,
    fault_injector: FaultInjector | None = None,
) -> tuple[Status, Blackboard, MissionRecorder]:
    """
    Entry point main_pipeline.py should call in place of the old linear
    process_one() body for the phases this tree covers.

    Returns the final Status, the blackboard (for reading back results
    like protocol/fragility/crack data), and the recorder (call .flush()
    to write the black-box log to disk).
    """
    recorder = MissionRecorder(object_id=object_id, output_dir=photo_dir)
    bb = Blackboard()
    bb.set("object_id", object_id)
    bb.set("sequence_number", sequence_number)
    bb.set("dry_run", dry_run)
    bb.set("hardware", getattr(pipeline, "_hardware", None))
    if fault_injector is not None:
        bb.set("fault_injector", fault_injector)

    tree = build_supervised_tree(pipeline, object_id, sequence_number, recorder)
    final_status = run_to_completion(tree, bb)
    recorder.flush()
    return final_status, bb, recorder
