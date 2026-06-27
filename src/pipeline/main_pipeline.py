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
from config import OBJ_CLASSES, SCAN_PHOTOS_DIR
from src.analysis.treatment_advisor import DiagnosticInputs, run_treatment_advisor
from src.catalogue.records import ObjectRecord
from src.catalogue.service import CatalogueService
from src.crack_detection.detector import CrackDetector
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
        self._emit("mission_started", {"total_artefacts": self.artefact_count})

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

        frame = self._capture_reference_frame(object_id)
        self._emit("scan_started", {"artefact_id": object_id})

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

        class_name = OBJ_CLASSES[(sequence_number - 1) % len(OBJ_CLASSES)]
        protocol = run_treatment_advisor(DiagnosticInputs(
            artefact_id=object_id,
            artefact_class=class_name,
            crack_severity=crack_result.severity_score,
            salt_risk=salt_result.overall_risk,
            salt_critical=len(salt_result.critical_zones) > 0,
            stress_score=0.2,
            ocr_confidence=0.0,
            has_inscription=class_name in {"coin", "inscription_fragment", "stone_carving"},
            biological_detected=False,
            years_remaining=30.0,
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

    def _pick_and_place(self) -> None:
        """Run the hardware pick/place sequence using the controller layer."""
        from src.arm.controller import ArduinoSerial, ArmController

        if self._hardware is None:
            arduino = ArduinoSerial()
            self._hardware = ArmController(arduino)
        self._hardware.pick_from_tray()
        self._hardware.place_on_turntable()

    def _capture_reference_frame(self, object_id: str) -> np.ndarray:
        """
        Capture or synthesize one reference frame for analysis stages.

        Args:
            object_id: Catalogue ID used for dry-run photo output.

        Returns:
            BGR image suitable for OpenCV modules.
        """
        if self.dry_run:
            frame = self._synthetic_artefact_frame()
            photo_dir = SCAN_PHOTOS_DIR / object_id
            photo_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(photo_dir / f"{object_id}_reference.jpg"), frame)
            return frame

        from src.arm.controller import CameraInterface

        camera = CameraInterface()
        photo_dir = SCAN_PHOTOS_DIR / object_id
        photo_dir.mkdir(parents=True, exist_ok=True)
        path = photo_dir / f"{object_id}_reference.jpg"
        camera.capture_frame(path)
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Captured image could not be read: {path}")
        return frame

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
