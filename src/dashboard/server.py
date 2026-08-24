# src/dashboard/server.py
"""
AXUM Rover — Dashboard Server
==============================
Flask-SocketIO backend that bridges the mission pipeline
to the frontend dashboard in real time.

Run: python src/dashboard/server.py
Then open: http://localhost:5000 in a browser

The mission pipeline emits events via emit_event().
The frontend receives them instantly via WebSocket.

Author: Axum Rover Team
"""

import base64
import json
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, render_template, send_from_directory, jsonify, abort
from flask_socketio import SocketIO, emit
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── App setup ─────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static")
)
app.config["SECRET_KEY"] = "axum_rover_2026"
app.config["TEMPLATES_AUTO_RELOAD"] = True  # pick up catalogue.html edits without restart
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Global mission state ──────────────────────────────────────
# This is the single source of truth for the dashboard.
# main_pipeline.py updates this dict; the frontend reads it.
mission_state = {
    "status":           "IDLE",        # IDLE | SCANNING | PROCESSING | COMPLETE
    "current_artefact": None,
    "artefacts_done":   0,
    "artefacts_total":  0,
    "log":              [],
    "catalogue":        [],
    "attitude":         None,          # last RoverAttitude payload, see emit_attitude
}

# Dashboard server instance (singleton)
_server_instance = None


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — EVENT EMITTERS
# Called from main_pipeline.py to push updates to the frontend
# ═══════════════════════════════════════════════════════════════

def emit_event(event_name: str, data: dict):
    """
    Emit a named event to all connected dashboard clients.

    This is the ONLY function main_pipeline.py needs to call.
    Import it and call it anywhere in the mission loop.

    Args:
        event_name: One of the event names listed below
        data:       Event payload dict

    Example:
        from src.dashboard.server import emit_event

        emit_event("crack_detected", {
            "severity":    0.73,
            "image_b64":   encode_image(crack_overlay),
            "artefact_id": "AX001"
        })

    Supported events:
        mission_started         — mission begins
        artefact_picked         — robot picked up an artefact
        scan_started            — scanning sequence started
        camera_frame            — live camera frame update
        crack_detected          — crack analysis complete
        stress_detected         — NDCI/multispectral result
        salt_detected           — salt mapping result
        depth_map_ready         — photometric stereo complete
        inscription_recognized  — OCR result
        translation_ready       — LLM translation complete
        fragility_clock         — years-remaining calculation
        treatment_protocol      — safe/unsafe intervention guardrail
        artefact_complete       — full artefact processed
        catalogue_updated       — new catalogue entry added
        attitude_update         — GY-80 IMU roll/pitch/yaw
        mission_complete        — all artefacts processed
        robot_status            — general status update
        error                   — error occurred
    """
    socketio.emit(event_name, data)
    _log_event(event_name, data)


def emit_attitude(roll: float, pitch: float, yaw: float, *,
                  is_valid: bool = True, gyro_biased: bool = True,
                  filter_hz: float = 0.0):
    """
    Push an orientation reading to the dashboard's attitude panel.

    Mirrors ``RoverAttitude`` from ``src/arm/controller.py`` field for
    field, so a caller polling the Arduino can forward a reading without
    reshaping it::

        att = arduino.attitude()
        emit_attitude(att.roll, att.pitch, att.yaw,
                      is_valid=att.is_valid,
                      gyro_biased=att.gyro_biased,
                      filter_hz=att.filter_hz)

    Args:
        roll:  Degrees, -180..180.
        pitch: Degrees, -90..90.
        yaw:   Degrees, 0..360 (the panel's compass rose assumes unsigned).
        is_valid: False when no IMU is fitted. The panel then shows "no
            signal" rather than drawing a level rover, which would be a
            actively misleading readout on a tilted vehicle.
        gyro_biased: False flags the reading as drifting in the UI.
        filter_hz: Observed firmware fusion rate.

    Rate: the firmware fuses at 100Hz but this is a WebSocket broadcast --
    push at 10-20Hz. The panel interpolates between frames, so faster only
    costs bandwidth.
    """
    payload = {
        "roll":        round(float(roll), 2),
        "pitch":       round(float(pitch), 2),
        # Wrapped here rather than trusting callers: the panel's compass
        # rose reads yaw as unsigned, and an out-of-range value renders as
        # a plausible-looking wrong heading instead of an obvious error.
        "yaw":         round(float(yaw) % 360.0, 2),
        "is_valid":    bool(is_valid),
        "gyro_biased": bool(gyro_biased),
        "filter_hz":   round(float(filter_hz), 1),
    }
    mission_state["attitude"] = payload
    emit_event("attitude_update", payload)


def emit_camera_frame(frame: np.ndarray, overlays: dict = None):
    """
    Emit a live camera frame to the dashboard.

    Call this in your main loop at ~5fps for smooth video.
    Higher rates will saturate the WebSocket.

    Args:
        frame:    BGR numpy array from ESP32-CAM
        overlays: Optional dict of overlay data:
            {
                "cracks":       list of (x,y,w,h) boxes,
                "text_regions": list of (x,y,w,h) boxes,
                "stress_zones": list of (x,y,w,h) boxes
            }
    """
    if frame is None:
        return

    # Draw overlays if provided
    display = frame.copy()
    if overlays:
        for x, y, w, h in overlays.get("cracks", []):
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 2)
        for x, y, w, h in overlays.get("text_regions", []):
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        for x, y, w, h in overlays.get("stress_zones", []):
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 165, 255), 2)

    emit_event("camera_frame", {
        "image_b64": encode_image(display, quality=60)
    })


def emit_inscription(text: str, translation: str, confidence: float,
                     is_new_discovery: bool = False, site: str = None):
    """
    Emit OCR inscription result with translation.

    Args:
        text:             Recognized Ge'ez text
        translation:      English translation
        confidence:       OCR confidence 0–1
        is_new_discovery: True if not in inscription database
        site:             Known site name if matched
    """
    emit_event("inscription_recognized", {
        "text":             text,
        "translation":      translation,
        "confidence":       confidence,
        "is_new_discovery": is_new_discovery,
        "site":             site,
        "chars":            list(text)  # for character-by-character animation
    })


def emit_treatment_protocol(protocol: dict):
    """
    Emit field treatment protocol (safe / unsafe interventions).

    Args:
        protocol: TreatmentProtocol.to_dict() from treatment_advisor
    """
    emit_event("treatment_protocol", protocol)
    log_message(
        f"Treatment protocol ready — urgency {protocol.get('urgency', 'UNKNOWN')}",
        "warning" if protocol.get("urgency_level", 0) >= 3 else "info",
    )


def emit_fragility_clock(artefact_id: str, years_remaining: float,
                         risk_factors: list = None):
    """
    Emit fragility clock calculation.

    Args:
        artefact_id:    Artefact identifier
        years_remaining: Estimated years before irreversible damage
        risk_factors:   List of strings describing risk factors
    """
    from config import TREATMENT_URGENCY_CRITICAL_YEARS

    emit_event("fragility_clock", {
        "artefact_id":    artefact_id,
        "years_remaining": round(years_remaining, 1),
        "risk_factors":   risk_factors or [],
        "is_critical":    years_remaining < TREATMENT_URGENCY_CRITICAL_YEARS,
    })


def emit_catalogue_entry(entry: dict):
    """
    Emit a completed catalogue entry.

    Args:
        entry: Dict with keys:
            id, classification, damage_score, inscription_text,
            translation, years_remaining, image_b64, timestamp
    """
    mission_state["catalogue"].append(entry)
    mission_state["artefacts_done"] += 1
    emit_event("catalogue_updated", {
        "entry":          entry,
        "total_complete": mission_state["artefacts_done"]
    })


def emit_resurrection_wall():
    """
    Trigger the Resurrection Wall full-screen display.
    Call this at mission end.
    """
    inscriptions = [
        e for e in mission_state["catalogue"]
        if e.get("inscription_text")
    ]
    emit_event("resurrection_wall", {
        "inscriptions": inscriptions,
        "total":        len(inscriptions)
    })


def log_message(message: str, level: str = "info"):
    """
    Add a message to the mission log (visible in dashboard).

    Args:
        message: Human-readable log message
        level:   "info" | "warning" | "success" | "error"
    """
    entry = {
        "message":   message,
        "level":     level,
        "timestamp": time.strftime("%H:%M:%S")
    }
    mission_state["log"].append(entry)
    if len(mission_state["log"]) > 100:
        mission_state["log"].pop(0)
    emit_event("log_message", entry)


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def encode_image(image: np.ndarray, quality: int = 85) -> str:
    """
    Encode a numpy image array to base64 JPEG string for WebSocket.

    Args:
        image:   BGR numpy array
        quality: JPEG quality 0–100 (lower = smaller = faster)

    Returns:
        Base64 encoded string, or empty string on failure
    """
    if image is None:
        return ""
    try:
        _, buffer = cv2.imencode(
            ".jpg", image,
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        logger.error(f"Image encode failed: {e}")
        return ""


def _log_event(event_name: str, data: dict):
    """Internal: log emitted events for debugging."""
    # Camera frames and attitude are continuous telemetry — at 5fps and
    # 15Hz respectively they bury every other line in the debug log.
    if event_name in ("camera_frame", "attitude_update"):
        return
    logger.debug(f"[WS] {event_name}: "
                f"{str(data)[:120]}{'...' if len(str(data)) > 120 else ''}")


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — FLASK ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the main dashboard."""
    from config import DASHBOARD_FEATURE_MESH
    return render_template("dashboard.html", feature_mesh=DASHBOARD_FEATURE_MESH)


@app.route("/catalogue")
def catalogue_viewer():
    """Serve the interactive 3D heritage catalogue viewer."""
    return render_template("catalogue.html")


@app.route("/state")
def get_state():
    """Return current mission state as JSON (for initial page load)."""
    return json.dumps(mission_state)


@app.route("/api/catalogue")
def api_catalogue():
    """Return all scanned objects as JSON for the catalogue viewer."""
    from src.catalogue.service import CatalogueService
    service = CatalogueService()
    return jsonify(service.load_all_objects())


@app.route("/api/artefact/<object_id>")
def api_artefact(object_id: str):
    """Return one catalogue entry by AXUM-OBJ-xxx ID."""
    from src.catalogue.service import CatalogueService
    service = CatalogueService()
    entry = service.get_object(object_id)
    if not entry:
        abort(404)
    return jsonify(entry)


@app.route("/api/fragment-groups")
def api_fragment_groups():
    """Return all vessel fragment groups with member records."""
    from src.analysis.fragment_grouper import FragmentGrouper
    grouper = FragmentGrouper()
    return jsonify({"groups": grouper.summary_for_api()})


@app.route("/api/mesh/<object_id>/status")
def api_mesh_status(object_id: str):
    """Return mesh readiness and validation info for the 3D viewer."""
    from src.catalogue.mesh_registry import get_mesh_status
    return jsonify(get_mesh_status(object_id))


@app.route("/api/mesh-preview/<object_id>")
def api_mesh_preview(object_id: str):
    """Return a catalogue-independent viewer entry for any published mesh folder."""
    from src.catalogue.mesh_registry import preview_mesh_entry
    entry = preview_mesh_entry(object_id)
    if not entry:
        abort(404)
    return jsonify(entry)


@app.route("/api/mesh-library")
def api_mesh_library():
    """
    Return published meshes that have no catalogue record.

    These are reconstructions like TEST-SCEAUX -- real Meshroom output
    used to validate the photogrammetry pipeline, but not heritage
    records. The catalogue grid appends them so the pipeline's actual
    output is visible alongside catalogued artefacts, flagged as
    ``is_preview`` so the UI can label them rather than passing them off
    as scanned finds.
    """
    from src.catalogue.mesh_registry import preview_mesh_entry, scan_published_meshes
    from src.catalogue.service import CatalogueService

    catalogued = {o.get("object_id") for o in CatalogueService().load_all_objects()}
    entries = []
    for status in scan_published_meshes():
        object_id = status.get("object_id")
        if not object_id or object_id in catalogued:
            continue
        entry = preview_mesh_entry(object_id)
        if entry:
            entries.append(entry)
    return jsonify(entries)


@app.route("/api/rover-views")
def api_rover_views():
    """
    Report which masked rover silhouettes are available to the attitude panel.

    Drop ``top.png``, ``front.png`` and ``side.png`` into
    ``src/dashboard/static/rover/`` and they are picked up on next page
    load. Any view without a file falls back to the built-in SVG
    silhouette, so a partial set is fine.
    """
    from config import ROVER_VIEW_DIR, ROVER_VIEW_EXTENSIONS

    views = {}
    for view in ("top", "front", "side"):
        for ext in ROVER_VIEW_EXTENSIONS:
            candidate = ROVER_VIEW_DIR / f"{view}{ext}"
            if candidate.is_file():
                views[view] = f"/static/rover/{candidate.name}"
                break
    return jsonify({"views": views, "dir": str(ROVER_VIEW_DIR)})


@app.route("/models/<object_id>/<path:filename>")
def serve_mesh(object_id: str, filename: str):
    """
    Serve published Meshroom assets (OBJ, MTL, textures) for the 3D viewer.

    All files for one object live in ``scans/meshes/<object_id>/``.
    """
    from config import MESH_DIR, SCANS_DIR

    search_roots = (
        MESH_DIR / object_id,
        SCANS_DIR / "meshes" / object_id,
        SCANS_DIR / object_id,
        MESH_DIR,
    )
    for base in search_roots:
        path = (base / filename).resolve()
        try:
            path.relative_to(base.resolve())
        except ValueError:
            continue
        if path.exists() and path.is_file():
            return send_from_directory(path.parent, path.name)
    abort(404)


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — SOCKETIO EVENT HANDLERS
# ═══════════════════════════════════════════════════════════════

@socketio.on("connect")
def on_connect():
    """Send current state to newly connected client."""
    logger.info("Dashboard client connected")
    emit("state_sync", mission_state)
    if mission_state["attitude"]:
        emit("attitude_update", mission_state["attitude"])
    emit("log_message", {
        "message":   "Dashboard connected — AXUM Rover ready",
        "level":     "success",
        "timestamp": time.strftime("%H:%M:%S")
    })


@socketio.on("disconnect")
def on_disconnect():
    logger.info("Dashboard client disconnected")


@socketio.on("request_state")
def on_request_state():
    """Client requesting full state refresh."""
    emit("state_sync", mission_state)


@socketio.on("trigger_resurrection_wall")
def on_trigger_resurrection():
    """Client-triggered Resurrection Wall (for demo control)."""
    emit_resurrection_wall()


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — SERVER LIFECYCLE
# ═══════════════════════════════════════════════════════════════

def start_server(host: str = "0.0.0.0", port: int = 5000,
                 open_browser: bool = True):
    """
    Start the dashboard server in a background thread.

    Call this from main_pipeline.py before starting the mission:

        from src.dashboard.server import start_server
        start_server()

    Args:
        host:         Host to bind (0.0.0.0 = all interfaces)
        port:         Port number
        open_browser: Auto-open browser on start
    """
    global _server_instance

    if open_browser:
        def _open_browser():
            time.sleep(1.5)
            import webbrowser
            webbrowser.open(f"http://localhost:{port}")
        threading.Thread(target=_open_browser, daemon=True).start()

    logger.info(f"Dashboard: http://localhost:{port}")

    # Run in background thread so mission pipeline can continue
    server_thread = threading.Thread(
        target=lambda: socketio.run(
            app, host=host, port=port,
            debug=False, use_reloader=False
        ),
        daemon=True
    )
    server_thread.start()
    _server_instance = server_thread
    time.sleep(1.0)  # wait for server to be ready
    return server_thread


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — DEMO / TEST MODE
# Runs without hardware to test the dashboard UI
# ═══════════════════════════════════════════════════════════════

def run_mission_in_background(
    artefact_count: int = 1,
    *,
    hardware: bool = False,
    generate_mesh: bool = True,
) -> threading.Thread:
    """
    Run MissionPipeline in a daemon thread inside this dashboard process.

    WHAT: Starts the real mission loop after the SocketIO server is live so
    ``emit_event()`` calls hit the same ``socketio`` instance browsers use.
    WHY: Importing dashboard emitters from a separate pipeline process creates
    a second SocketIO with no subscribers — the root cause of silent live
    dashboard failures.

    Args:
        artefact_count: Number of artefacts to process.
        hardware:       When True, open serial/camera hardware (not dry-run).
        generate_mesh:  When False, skip mesh stage.

    Returns:
        The started daemon thread (already running).
    """
    def _runner() -> None:
        from src.pipeline.main_pipeline import MissionPipeline

        mission_state["status"] = "SCANNING"
        mission_state["artefacts_total"] = artefact_count
        mission_state["artefacts_done"] = 0
        log_message(
            f"Mission started ({'hardware' if hardware else 'dry-run'})",
            "info",
        )
        try:
            pipeline = MissionPipeline(
                artefact_count=artefact_count,
                dry_run=not hardware,
                generate_mesh=generate_mesh,
                emit_dashboard=True,
            )
            records = pipeline.run()
            mission_state["status"] = "COMPLETE"
            log_message(
                f"Mission complete — {len(records)} artefact(s) processed",
                "success",
            )
        except Exception as exc:
            mission_state["status"] = "ERROR"
            log_message(f"Mission failed: {exc}", "error")
            logger.exception("Mission thread failed")

    thread = threading.Thread(target=_runner, daemon=True, name="axum-mission")
    thread.start()
    return thread


def start_attitude_simulation(*, rate_hz: float = 15.0) -> threading.Thread:
    """
    Drive the attitude panel with synthetic terrain motion.

    For UI work and demos only -- this is NOT a model of the rover's
    dynamics and must never run alongside a real IMU feed, or the panel
    will show whichever emitter wrote last. Real telemetry path is
    ``ArduinoSerial.attitude()`` -> ``emit_attitude()``.

    Reports ``gyro_biased=True`` because a simulated signal has no drift;
    the drift indicator is meaningful only for real hardware.

    Returns:
        The daemon thread, already started.
    """
    import math
    import random

    def _runner() -> None:
        t = 0.0
        period = 1.0 / rate_hz
        heading = random.uniform(0, 360)
        while True:
            t += period
            # Superimposed periods so no two axes look synchronised --
            # a single shared frequency reads obviously fake.
            roll = 6.5 * math.sin(t * 0.62) + 1.8 * math.sin(t * 2.7)
            pitch = 8.0 * math.sin(t * 0.41 + 1.1) + 1.2 * math.sin(t * 3.3)
            heading = (heading + 0.35 * math.sin(t * 0.23)) % 360.0
            emit_attitude(roll, pitch, heading, filter_hz=100.0)
            time.sleep(period)

    thread = threading.Thread(target=_runner, daemon=True, name="axum-attitude-sim")
    thread.start()
    return thread


def run_demo_sequence():
    """
    Simulate a complete mission for UI testing.
    Run: python src/dashboard/server.py --demo
    """
    import random

    time.sleep(2)  # wait for browser to open
    logger.info("Starting demo sequence...")

    start_attitude_simulation()

    log_message("AXUM Rover initializing...", "info")
    time.sleep(0.5)

    emit_event("mission_started", {
        "total_artefacts": 3,
        "mission_id":      "DEMO_001"
    })
    mission_state["status"] = "SCANNING"
    mission_state["artefacts_total"] = 3

    for i in range(1, 4):
        artefact_id = f"AX{i:03d}"
        log_message(f"Processing artefact {i}/3: {artefact_id}", "info")

        emit_event("artefact_picked", {
            "artefact_id": artefact_id,
            "position":    i
        })
        time.sleep(1)

        emit_event("scan_started", {"artefact_id": artefact_id})
        time.sleep(0.5)

        # Simulate crack detection
        severity = round(random.uniform(0.3, 0.9), 2)
        log_message(f"Crack detected — severity {severity}", "warning")
        emit_event("crack_detected", {
            "artefact_id": artefact_id,
            "severity":    severity,
            "crack_count": random.randint(1, 5),
            "image_b64":   ""
        })
        time.sleep(1)

        # Simulate stress detection
        stress = round(random.uniform(0.1, 0.6), 2)
        emit_event("stress_detected", {
            "artefact_id":   artefact_id,
            "overall_stress": stress,
            "risk_zones":    random.randint(0, 4),
            "image_b64":     ""
        })
        time.sleep(0.8)

        # Simulate salt detection
        emit_event("salt_detected", {
            "artefact_id":  artefact_id,
            "overall_risk": round(random.uniform(0.0, 0.4), 2),
            "critical":     random.choice([True, False]),
            "image_b64":    ""
        })
        time.sleep(0.8)

        # Simulate inscription
        sample_texts = [
            ("ሰላም ለኪ", "Peace to you", 0.87),
            ("እግዚእነ ኢየሱስ ክርስቶስ", "Our Lord Jesus Christ", 0.92),
            ("ዓጼ ዳዊት", "Emperor David", 0.79),
        ]
        text, translation, conf = sample_texts[i - 1]
        is_new = i == 3  # third artefact is a "discovery"

        log_message(
            f"Inscription recognized: {text} ({conf:.0%})",
            "success" if not is_new else "warning"
        )
        emit_inscription(
            text, translation, conf,
            is_new_discovery=is_new,
            site="Lalibela, Beta Maryam" if not is_new else None
        )
        time.sleep(1.5)

        # Fragility clock
        years = round(random.uniform(5, 45), 1)
        emit_fragility_clock(artefact_id, years, [
            "Active crack near inscription",
            "Salt migration detected",
            "UV stress zones present"
        ])
        time.sleep(0.5)

        # Treatment protocol (decision guardrail)
        try:
            from src.analysis.treatment_advisor import DiagnosticInputs, run_treatment_advisor
            protocol = run_treatment_advisor(DiagnosticInputs(
                artefact_id=artefact_id,
                artefact_class=random.choice(
                    ["stone_carving", "pottery", "inscription_fragment"]
                ),
                crack_severity=severity,
                salt_risk=round(random.uniform(0.2, 0.8), 2),
                salt_critical=random.choice([True, False]),
                stress_score=stress,
                ocr_confidence=conf,
                has_inscription=bool(text),
                biological_detected=random.choice([True, False]),
                years_remaining=years,
                active_moisture=random.choice([True, False]),
            ))
            emit_treatment_protocol(protocol.to_dict())
        except Exception as e:
            logger.warning(f"Treatment advisor unavailable in demo: {e}")
        time.sleep(0.8)

        # Catalogue entry
        emit_catalogue_entry({
            "id":               artefact_id,
            "classification":   random.choice(
                ["stone_carving", "pottery", "inscription_fragment"]
            ),
            "damage_score":     severity,
            "inscription_text": text,
            "translation":      translation,
            "years_remaining":  years,
            "image_b64":        "",
            "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S")
        })
        time.sleep(1)

        log_message(f"Artefact {artefact_id} complete", "success")

    # Mission complete
    mission_state["status"] = "COMPLETE"
    emit_event("mission_complete", {
        "total_processed": 3,
        "inscriptions_found": 3,
        "new_discoveries": 1
    })
    log_message("Mission complete — 3 artefacts processed", "success")

    time.sleep(2)
    emit_resurrection_wall()


if __name__ == "__main__":
    import sys

    from config import DASHBOARD_HOST, DASHBOARD_PORT

    demo_mode = "--demo" in sys.argv
    mission_mode = "--mission" in sys.argv
    hardware_mode = "--hardware" in sys.argv
    count = 1
    for i, arg in enumerate(sys.argv):
        if arg == "--count" and i + 1 < len(sys.argv):
            try:
                count = int(sys.argv[i + 1])
            except ValueError:
                pass

    if demo_mode and mission_mode:
        print("Use either --demo or --mission, not both.")
        raise SystemExit(2)

    if demo_mode:
        print("Starting AXUM Dashboard in DEMO mode...")
        threading.Thread(target=run_demo_sequence, daemon=True).start()
    elif mission_mode:
        print(
            f"Starting AXUM Dashboard + mission "
            f"({'hardware' if hardware_mode else 'dry-run'}, count={count})..."
        )

        def _delayed_mission() -> None:
            time.sleep(2)
            run_mission_in_background(
                count, hardware=hardware_mode, generate_mesh=True
            )

        threading.Thread(target=_delayed_mission, daemon=True).start()

    print(f"Open http://localhost:{DASHBOARD_PORT} in your browser")
    socketio.run(
        app,
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        debug=False,
        use_reloader=False,
    )