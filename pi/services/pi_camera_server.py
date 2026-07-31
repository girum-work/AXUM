#!/usr/bin/env python3
"""
AXUM Rover — Pi 4 IR-CUT Camera Server
========================================
WHAT:  Runs on the Raspberry Pi 4 and exposes the same HTTP API as the
       original ESP32-CAM: /capture (JPEG still) and /stream (MJPEG).
       The laptop pipeline (CameraInterface in controller.py) connects here
       exactly as it connected to the ESP32-CAM — no changes needed there.
WHY:   Keeps CameraInterface's API stable across camera hardware changes.
       The Pi IR-CUT module gives higher resolution, adjustable focus, and
       built-in IR illumination for low-light scanning.

DEPLOY: Copy this file to the Pi, install dependencies, then run:
    pip install flask picamera2
    python3 pi_camera_server.py

UPDATE config.py on the laptop:
    PI_CAM_URL     = "http://<pi-ip>:5001/stream"
    PI_CAM_CAPTURE = "http://<pi-ip>:5001/capture"

HARDWARE: Raspberry Pi 4 + Pi IR-CUT Night Vision Camera Module (OV5647).
    The IR-CUT filter is controlled via GPIO — HIGH = daylight, LOW = IR mode.
    Default: daylight mode (IR-CUT filter in).
"""

import io
import time
import threading
from pathlib import Path

from flask import Flask, Response, jsonify, request

# picamera2 is only available on the Pi — import is guarded for development
try:
    from picamera2 import Picamera2
    from picamera2.encoders import MJPEGEncoder
    from picamera2.outputs import FileOutput
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────
CAPTURE_WIDTH   = 1280
CAPTURE_HEIGHT  = 960
STREAM_WIDTH    = 640
STREAM_HEIGHT   = 480
STREAM_FPS      = 10
IR_CUT_PIN      = 17    # BCM pin controlling the IR-CUT filter relay
SERVER_PORT     = 5001  # must match PI_CAM_URL port in laptop config.py

app = Flask(__name__)

# ── Camera singleton ──────────────────────────────────────────
_camera: "Picamera2 | None" = None
_camera_lock = threading.Lock()
_stream_frame: bytes = b""
_stream_lock  = threading.Lock()


def get_camera() -> "Picamera2":
    """
    Initialise and return the shared Picamera2 instance.

    WHY singleton: picamera2 does not allow multiple open instances; all
    routes must share one object.
    """
    global _camera
    if _camera is None:
        if not PICAMERA_AVAILABLE:
            raise RuntimeError(
                "picamera2 not available — run this script on a Raspberry Pi 4"
            )
        cam = Picamera2()
        # Main (still) configuration: full resolution
        still_cfg  = cam.create_still_configuration(
            main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "BGR888"},
        )
        cam.configure(still_cfg)
        cam.start()
        time.sleep(1.0)   # allow auto-exposure to settle
        _camera = cam
    return _camera


# ── IR-CUT filter control ─────────────────────────────────────

def _setup_gpio() -> None:
    """Set up GPIO for IR-CUT filter relay if available."""
    if GPIO_AVAILABLE:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(IR_CUT_PIN, GPIO.OUT)
        GPIO.output(IR_CUT_PIN, GPIO.HIGH)   # default: daylight (filter IN)


def set_ir_mode(ir_on: bool) -> None:
    """
    Switch between IR and daylight mode.

    Args:
        ir_on: True = IR mode (filter OUT, see in dark with IR LEDs).
               False = daylight mode (filter IN, accurate colour).
    """
    if GPIO_AVAILABLE:
        GPIO.output(IR_CUT_PIN, GPIO.LOW if ir_on else GPIO.HIGH)


# ── Streaming thread ──────────────────────────────────────────

def _stream_worker() -> None:
    """
    Background thread: continuously captures MJPEG frames into _stream_frame.

    WHY separate thread: keeps /stream response latency low regardless of
    /capture calls hitting the same camera.
    """
    global _stream_frame
    cam = get_camera()
    # Reconfigure for lower-res video (keeps CPU load manageable on Pi 4)
    video_cfg = cam.create_video_configuration(
        main={"size": (STREAM_WIDTH, STREAM_HEIGHT), "format": "BGR888"},
        controls={"FrameRate": STREAM_FPS},
    )
    cam.configure(video_cfg)
    cam.start()

    while True:
        buf = io.BytesIO()
        cam.capture_file(buf, format="jpeg")
        buf.seek(0)
        with _stream_lock:
            _stream_frame = buf.read()
        time.sleep(1.0 / STREAM_FPS)


# ── Routes ────────────────────────────────────────────────────

@app.route("/capture")
def capture() -> Response:
    """
    Capture one JPEG still at full resolution.

    Returns:
        JPEG image with Content-Type image/jpeg.
        Compatible with CameraInterface.capture_frame() in controller.py.
    """
    with _camera_lock:
        cam = get_camera()
        buf = io.BytesIO()
        cam.capture_file(buf, format="jpeg")
        buf.seek(0)
        jpeg = buf.read()

    return Response(jpeg, mimetype="image/jpeg")


def _mjpeg_generator():
    """Yield MJPEG multipart frames from the background stream thread."""
    while True:
        with _stream_lock:
            frame = _stream_frame
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame +
                b"\r\n"
            )
        time.sleep(1.0 / STREAM_FPS)


@app.route("/stream")
def stream() -> Response:
    """
    MJPEG stream endpoint.

    Compatible with CameraInterface.stream_endpoint() in controller.py.
    Point a browser or OpenCV VideoCapture at this URL to preview.
    """
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/ir", methods=["POST"])
def ir_mode() -> Response:
    """
    Switch IR-CUT filter.

    Body JSON: {"ir": true}  → IR mode (dark scanning)
               {"ir": false} → daylight mode (colour accurate)
    """
    data = request.get_json(force=True, silent=True) or {}
    ir_on = bool(data.get("ir", False))
    set_ir_mode(ir_on)
    return jsonify({"ok": True, "ir_mode": ir_on})


@app.route("/status")
def status() -> Response:
    """Health check — laptop pipeline calls this to confirm the server is up."""
    return jsonify({
        "ok": True,
        "picamera_available": PICAMERA_AVAILABLE,
        "gpio_available": GPIO_AVAILABLE,
        "resolution": f"{CAPTURE_WIDTH}x{CAPTURE_HEIGHT}",
        "stream_resolution": f"{STREAM_WIDTH}x{STREAM_HEIGHT}",
    })


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    _setup_gpio()
    if PICAMERA_AVAILABLE:
        # Pre-warm camera before accepting requests
        get_camera()
        # Start background stream thread
        t = threading.Thread(target=_stream_worker, daemon=True)
        t.start()
        print(f"AXUM Pi Camera Server running on port {SERVER_PORT}")
        print(f"  Still capture : http://<pi-ip>:{SERVER_PORT}/capture")
        print(f"  MJPEG stream  : http://<pi-ip>:{SERVER_PORT}/stream")
        print(f"  IR mode toggle: POST http://<pi-ip>:{SERVER_PORT}/ir")
    else:
        print("WARNING: picamera2 not found — running in stub mode (status only)")

    app.run(host="0.0.0.0", port=SERVER_PORT, threaded=True)