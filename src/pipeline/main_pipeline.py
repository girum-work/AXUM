"""
AXUM ROVER - Mission orchestration.

WHAT: A small but functional mission loop that ties dashboard events,
analysis modules, catalogue records, and optional hardware together.
WHY: The project needs an executable integration point before the robot is
fully calibrated; dry-run mode proves the data flow without touching motors.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import COMPUTE_TIER, FIRMWARE_LED_READY, OBJ_CLASSES, SCAN_PHOTOS_DIR, TURNTABLE_STEPS
from src.analysis.fragility_clock import FragilityInputs, run_fragility_clock
from src.analysis.treatment_advisor import DiagnosticInputs, run_treatment_advisor
from src.catalogue.records import ObjectRecord
from src.catalogue.service import CatalogueService
from src.crack_detection.detector import CrackDetector
from src.imaging.multispectral import analyse_multispectral_paths
from src.imaging.photometric_stereo import analyse_photometric_paths
from src.imaging.salt_mapper import run_salt_mapper
from src.pipeline.mesh_stage import process_object_mesh

@dataclass
class MissionRecord:
    """
    Summary for one processed artefact.

    Args:
        object_id: Catalogue ID assigned during the mission.
        status: Processing status for dashboards and logs.
        errors: Non-fatal warnings collected during processing.
    """

    object_id: str
    status: str = "pending"
    errors: list[str] = field(default_factory=list)


@dataclass
class MissionState:
    """
    Mutable mission progress state shared with callers.

    Args:
        status: Overall mission status.
        artefacts_total: Number of artefacts requested.
        artefacts_done: Number of artefacts completed.
        current_artefact: Current artefact ID, if any.
        records: Per-object mission records.
    """

    status: str = "IDLE"
    artefacts_total: int = 0
    artefacts_done: int = 0
    current_artefact: str | None = None
    records: list[MissionRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot of mission state."""
        return {
            "status": self.status,
            "artefacts_total": self.artefacts_total,
            "artefacts_done": self.artefacts_done,
            "current_artefact": self.current_artefact,
            "records": [r.__dict__ for r in self.records],
        }


SharedState = MissionState
state = MissionState()


class MissionPipeline:
    """
    End-to-end mission pipeline with safe dry-run defaults.

    Args:
        artefact_count: Number of artefacts to process.
        dry_run: When True, do not open serial ports or require the ESP32-CAM.
        generate_mesh: When True, publish/generate mesh data for each object.
        emit_dashboard: When True, send events to the Flask dashboard module.
    """

    def __init__(
        self,
        artefact_count: int = 1,
        *,
        dry_run: bool = True,
        generate_mesh: bool = True,
        emit_dashboard: bool = True,
    ) -> None:
        self.artefact_count = artefact_count
        self.dry_run = dry_run
        self.generate_mesh = generate_mesh
        self.emit_dashboard = emit_dashboard
        self.state = state
        self.catalogue = CatalogueService()
        self.crack_detector = CrackDetector()
        self._hardware = None
        # Read once, here, not per-inference-call. See CTO's Friday GPU
        # studio directive — this is intentionally a startup-time decision,
        # not something re-checked mid-mission. Any inference-owning module
        # (classifier, OCR, LLM restoration) should import COMPUTE_TIER
        # from config.py directly rather than expect it threaded through
        # as a parameter from here — this pipeline doesn't run inference
        # itself, it orchestrates modules that do, so there's no reason to
        # add coupling by passing this value through call sites it doesn't
        # otherwise need to touch.
        self.compute_tier = COMPUTE_TIER
        logger.info(f"MissionPipeline starting on compute tier: {self.compute_tier}")

    def run(self) -> list[ObjectRecord]:
        """
        Process every requested artefact and return catalogue records.

        Dry-run mode creates synthetic sensor inputs but still exercises the
        same catalogue, treatment, dashboard, and mesh code paths.
        """
        self.state.status = "SCANNING"
        self.state.artefacts_total = self.artefact_count
        self.state.artefacts_done = 0
        self.state.records.clear()
        self._emit("mission_started", {"total_artefacts": self.artefact_count, "compute_tier": self.compute_tier})

        outputs: list[ObjectRecord] = []
        for sequence in range(1, self.artefact_count + 1):
            object_id = f"AXUM-OBJ-{sequence:03d}"
            self.state.current_artefact = object_id
            mission_record = MissionRecord(object_id=object_id, status="processing")
            self.state.records.append(mission_record)
            try:
                record = self.process_one(object_id, sequence)
                outputs.append(record)
                mission_record.status = "complete"
                self.state.artefacts_done += 1
                self._emit("artefact_complete", record.to_dict())
            except Exception as exc:
                mission_record.status = "error"
                mission_record.errors.append(str(exc))
                logger.exception(f"Mission failed for {object_id}")
                self._emit("error", {"artefact_id": object_id, "message": str(exc)})

        self.state.status = "COMPLETE"
        self.state.current_artefact = None
        self._emit("mission_complete", self.state.to_dict())
        return outputs

    def process_one(self, object_id: str, sequence_number: int) -> ObjectRecord:
        """
        Process one artefact through scan, analysis, treatment, mesh, catalogue.

        Args:
            object_id: Catalogue ID for this artefact.
            sequence_number: Mission order number.

        Returns:
            Registered ObjectRecord.
        """
        self._emit("artefact_picked", {"artefact_id": object_id, "dry_run": self.dry_run})
        if not self.dry_run:
            self._pick_and_place()

        self._emit("scan_started", {"artefact_id": object_id})
        scan = self._scan_artefact(object_id)
        frame = scan["reference_frame"]
        self._emit("scan_complete", {
            "artefact_id": object_id,
            "mesh_photo_count": scan["mesh_photo_count"],
            "led_ready": FIRMWARE_LED_READY or self.dry_run,
        })

        crack_result = self.crack_detector.detect(frame)
        self._emit("crack_detected", crack_result.to_dict())

        uv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        salt_result = run_salt_mapper(
            uv_image=uv_frame,
            visible_image=frame,
            artefact_id=object_id,
        )
        salt_stage = self._salt_stage_label(salt_result.overall_risk)
        self._emit("salt_detected", {
            "artefact_id": object_id,
            "overall_risk": salt_result.overall_risk,
            "critical": len(salt_result.critical_zones) > 0,
            "risk_level": salt_stage,
        })

        ps_result = analyse_photometric_paths(scan["quad_paths"])
        self._emit("photometric_stereo_complete", {
            "artefact_id": object_id,
            "quality_score": ps_result.quality_score,
            "depth_units": ps_result.depth_units,
        })

        ms_result = analyse_multispectral_paths(scan["uv_path"], scan["nir_path"])
        self._emit("multispectral_complete", {
            "artefact_id": object_id,
            "structural_hazard_score": ms_result.structural_hazard_score,
            "ndci_mean": float(ms_result.ndci_map.mean()),
        })

        class_name = OBJ_CLASSES[(sequence_number - 1) % len(OBJ_CLASSES)]
        has_inscription = class_name in {"coin", "inscription_fragment", "stone_carving"}

        fragility = run_fragility_clock(FragilityInputs(
            artefact_id=object_id,
            artefact_class=class_name,
            crack_severity=crack_result.severity_score,
            salt_risk=salt_result.overall_risk,
            salt_critical=len(salt_result.critical_zones) > 0,
            stress_score=ms_result.structural_hazard_score,
            active_moisture=False,
        ))
        self._emit_fragility_clock(object_id, fragility, ms_result.structural_hazard_score)

        protocol = run_treatment_advisor(DiagnosticInputs(
            artefact_id=object_id,
            artefact_class=class_name,
            crack_severity=crack_result.severity_score,
            salt_risk=salt_result.overall_risk,
            salt_critical=len(salt_result.critical_zones) > 0,
            stress_score=ms_result.structural_hazard_score,
            ocr_confidence=0.0,
            has_inscription=has_inscription,
            biological_detected=False,
            years_remaining=fragility.years_remaining,
            active_moisture=False,
        ))
        self._emit("treatment_protocol", protocol.to_dict())

        record = ObjectRecord(
            object_id=object_id,
            sequence_number=sequence_number,
            class_name=class_name,
            class_confidence=1.0 if self.dry_run else 0.0,
            class_source="dry_run" if self.dry_run else "pipeline",
            crack_severity=crack_result.severity_score,
            salt_stage=salt_stage,
            photo_paths=self._photo_paths_for(object_id),
            photo_count=len(self._photo_paths_for(object_id)),
            interventions=[p.name for p in protocol.safe_treatments[:3]],
        )

        registered = self.catalogue.register_object(record)
        if self.generate_mesh:
            try:
                _mesh_result, mesh_record = process_object_mesh(
                    object_id,
                    skip_meshroom=self.dry_run,
                    register=True,
                )
                if mesh_record:
                    registered = mesh_record
            except Exception as exc:
                registered.errors.append(f"Mesh stage failed: {exc}")
                logger.warning(f"Mesh stage failed for {object_id}: {exc}")

        self._emit("catalogue_updated", {"entry": registered.to_dict()})
        return registered

    @staticmethod
    def _salt_stage_label(overall_risk: float) -> str:
        """Map numeric salt risk to a compact catalogue label."""
        if overall_risk >= 0.75:
            return "critical"
        if overall_risk >= 0.50:
            return "high"
        if overall_risk >= 0.25:
            return "moderate"
        if overall_risk > 0.05:
            return "low"
        return "none"

    def _ensure_hardware(self) -> "ArmController":
        """
        Lazily create the shared ArmController/ArduinoSerial connection.

        WHY this exists as one method instead of being inlined wherever
        hardware is needed: it used to be copy-pasted in _pick_and_place()
        and _scan_artefact() separately, and mission_tree.py's phase
        actions assumed one of those had already run before they did —
        which isn't true when PICK is the first phase to touch hardware.
        Self-review finding: on a fresh live-mode run, PICK ran before
        anything had lazily created _hardware, so it failed with an
        AttributeError that looked like a hardware fault instead of an
        initialization-ordering bug. Single method, single source of
        truth, callable from anywhere (including mission_tree.py) so the
        first phase that needs hardware — whichever one that ends up
        being — always gets a real, initialized controller.
        """
        if self._hardware is None:
            from src.arm.controller import ArduinoSerial, ArmController

            arduino = ArduinoSerial()
            self._hardware = ArmController(arduino)
        return self._hardware

    def _pick_and_place(self) -> None:
        """Run the hardware pick/place sequence using the controller layer."""
        hardware = self._ensure_hardware()
        hardware.pick_from_tray()
        hardware.place_on_turntable()

    def _scan_artefact(self, object_id: str) -> dict[str, Any]:
        """
        Run the full artefact scan: turntable multi-angle mesh set, plus
        photometric-stereo quadrant images and UV/NIR multispectral images.

        WHY: process_one() previously took a single reference photo and
        called that "the scan." process_object_mesh() needs a full rotation
        set to actually mesh anything, and photometric_stereo/multispectral
        need their own dedicated captures. This is one stage so all three
        downstream analyses read from the same physical scan pass instead
        of re-triggering hardware three separate times.

        Args:
            object_id: Catalogue ID used for output folder naming.

        Returns:
            Dict with reference_frame (ndarray), mesh_photo_count (int),
            quad_paths (dict[str, Path] for photometric stereo), and
            uv_path / nir_path (Path | None for multispectral).
        """
        photo_dir = SCAN_PHOTOS_DIR / object_id
        photo_dir.mkdir(parents=True, exist_ok=True)

        if self.dry_run:
            reference_frame = self._synthetic_artefact_frame()
            # HONESTY NOTE: previously simulated a multi-tilt-band capture
            # (TURNTABLE_TILT_BANDS x 36 photos) — but the real
            # TurntableController only has capture_rotation_set(), a single
            # flat 360° rotation, no tilt sweep (CAMERA_TILT isn't
            # implemented in firmware, and no controller method for it
            # exists either). Dry-run should exercise the SAME shape as
            # live, or it's testing against a capability that doesn't
            # exist — matching real behavior now: one tilt band,
            # TURNTABLE_STEPS photos (imported from config, not
            # hardcoded, so this can't silently drift from the real
            # firmware-facing constant).
            mesh_count = 0
            for index in range(TURNTABLE_STEPS):
                path = photo_dir / f"{object_id}_{index:03d}.jpg"
                cv2.imwrite(str(path), reference_frame)
                mesh_count += 1
            ps_dir = photo_dir / "ps"
            ps_dir.mkdir(exist_ok=True)
            quad_paths = {}
            for quad in ("N", "E", "S", "W"):
                path = ps_dir / f"{object_id}_ps_{quad}.jpg"
                cv2.imwrite(str(path), reference_frame)
                quad_paths[quad] = path
            ms_dir = photo_dir / "ms"
            ms_dir.mkdir(exist_ok=True)
            uv_path = ms_dir / f"{object_id}_uv.jpg"
            nir_path = ms_dir / f"{object_id}_nir.jpg"
            cv2.imwrite(str(uv_path), reference_frame)
            cv2.imwrite(str(nir_path), reference_frame)
            return {
                "reference_frame": reference_frame,
                "mesh_photo_count": mesh_count,
                "quad_paths": quad_paths,
                "uv_path": uv_path,
                "nir_path": nir_path,
            }

        if not FIRMWARE_LED_READY:
            raise RuntimeError(
                "Controlled lighting has not passed bench verification; "
                "refusing to create photometric or multispectral results."
            )

        raise RuntimeError(
            "Live scan requires an aligned visible/IR capture source. "
            "The current controller exposes only still-image capture, so "
            "the multispectral stage cannot be run honestly yet."
        )

        from config import PI_CAM_URL
        from src.arm.controller import CameraInterface, TurntableController

        arduino = self._ensure_hardware().arduino
        # CameraInterface defaults to ESP32_CAM_URL (its original design —
        # confirmed by reading the real controller.py, not assumed). The
        # scan stage needs the Pi IR-CUT camera specifically, so the URL
        # must be passed explicitly — relying on the default here was a
        # real bug, caught by checking against the actual file rather than
        # my earlier historical reference copy.
        camera = CameraInterface(stream_url=PI_CAM_URL, require_status=True)
        turntable = TurntableController(arduino=arduino, camera=camera)

        # HONESTY NOTE: TurntableController only has capture_rotation_set()
        # (single tilt band) in the real codebase — capture_multi_angle_set()
        # never existed; I was building against an assumption, not a
        # verified API. TURNTABLE_TILT_BANDS multi-angle sweep genuinely
        # cannot happen yet regardless — CAMERA_TILT isn't implemented in
        # axum_rover.ino, so there's no way to physically change tilt
        # between bands even if the method existed. Using the real method,
        # single tilt band, until both the controller method AND the
        # firmware command exist. Not pretending capability that isn't there.
        mesh_frames = turntable.capture_rotation_set(object_id, output_dir=photo_dir)
        if not mesh_frames:
            raise RuntimeError(f"Turntable scan produced zero frames for {object_id}")
        reference_frame = cv2.imread(str(mesh_frames[0].path))
        if reference_frame is None:
            raise RuntimeError(f"Captured image could not be read: {mesh_frames[0].path}")

        quad_paths = capture_quad_images(arduino, camera, object_id, output_dir=photo_dir / "ps")
        uv_path = capture_uv_image(arduino, camera, object_id, output_dir=photo_dir / "ms")
        nir_path = capture_nir_image(arduino, camera, object_id, output_dir=photo_dir / "ms")

        return {
            "reference_frame": reference_frame,
            "mesh_photo_count": len(mesh_frames),
            "quad_paths": quad_paths,
            "uv_path": uv_path,
            "nir_path": nir_path,
        }

    @staticmethod
    def _synthetic_artefact_frame() -> np.ndarray:
        """Create a stone-like image with a dark crack for dry-run analysis."""
        image = np.full((480, 640, 3), (160, 150, 130), dtype=np.uint8)
        cv2.ellipse(image, (320, 240), (170, 105), 0, 0, 360, (120, 112, 95), -1)
        cv2.line(image, (230, 210), (310, 250), (35, 35, 35), 3)
        cv2.line(image, (310, 250), (405, 235), (30, 30, 30), 2)
        cv2.circle(image, (380, 210), 18, (230, 230, 220), -1)
        noise = np.random.default_rng(42).normal(0, 7, image.shape).astype(np.int16)
        return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def _photo_paths_for(object_id: str) -> list[str]:
        """Return saved scan photo paths for an object ID."""
        photo_dir = SCAN_PHOTOS_DIR / object_id
        if not photo_dir.exists():
            return []
        return [str(path) for path in sorted(photo_dir.glob("*.jpg"))]

    def _emit(self, event_name: str, payload: dict[str, Any]) -> None:
        """Emit a dashboard event when enabled, logging failures as warnings."""
        if not self.emit_dashboard:
            return
        try:
            from src.dashboard.server import emit_event

            emit_event(event_name, payload)
        except Exception as exc:
            logger.debug(f"Dashboard event skipped ({event_name}): {exc}")

    def _emit_fragility_clock(self, object_id: str, fragility, stress_score: float) -> None:
        """
        Push a fragility-clock reading to the dashboard's typed endpoint.

        WHY a dedicated call instead of the generic _emit(): the dashboard's
        fragility widget (pulsing countdown) reads a fixed-shape function
        signature rather than a free-form payload dict.

        Signature confirmed against src/dashboard/server.py source (not an
        audit summary) as of this cycle:
            emit_fragility_clock(artefact_id, years_remaining, risk_factors=None)
        `stress_score` has no parameter in the deployed function — kept as
        an argument here only so callers/logs still have it, but it is not
        forwarded. If the dashboard should show it, that's a signature
        change on the Backend & Dashboard Engineer's side, not mine.

        Args:
            object_id: Catalogue ID (passed as artefact_id downstream).
            fragility: FragilityResult from run_fragility_clock().
            stress_score: Multispectral stress score. Not currently sent —
                no field to carry it in the deployed emit_fragility_clock.
        """
        if not self.emit_dashboard:
            return
        try:
            from src.dashboard.server import emit_fragility_clock

            emit_fragility_clock(
                artefact_id=object_id,
                years_remaining=fragility.years_remaining,
                risk_factors=[fragility.urgency_band] if fragility.urgency_band else None,
            )
        except Exception as exc:
            logger.debug(f"Dashboard fragility event skipped ({object_id}): {exc}")


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for dry-run and hardware missions.

    Args:
        argv: Optional argument list for tests.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Run AXUM mission pipeline")
    parser.add_argument("--count", type=int, default=1, help="artefacts to process")
    parser.add_argument("--hardware", action="store_true", help="use Arduino/ESP32 hardware")
    parser.add_argument("--no-mesh", action="store_true", help="skip mesh stage")
    parser.add_argument("--no-dashboard", action="store_true", help="do not emit dashboard events")
    args = parser.parse_args(argv)

    pipeline = MissionPipeline(
        artefact_count=args.count,
        dry_run=not args.hardware,
        generate_mesh=not args.no_mesh,
        emit_dashboard=not args.no_dashboard,
    )
    records = pipeline.run()
    logger.success(f"Mission complete: {len(records)} records")
    return 0 if records or args.count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())