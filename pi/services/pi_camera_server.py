#!/usr/bin/env python3
"""AXUM Raspberry Pi camera capture and preview service."""

from __future__ import annotations

import atexit
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import cv2
from flask import Flask, Response, jsonify

try:
    from picamera2 import Picamera2

    PICAMERA_AVAILABLE = True
except ImportError:
    Picamera2 = None  # type: ignore[assignment]
    PICAMERA_AVAILABLE = False

LOGGER = logging.getLogger("axum.pi.camera")

def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value

def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw in (None, "", "auto", "AUTO"):
        return None
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be positive or 'auto'")
    return value

@dataclass(frozen=True)
class CameraSettings:
    """Runtime settings supplied through the systemd environment file."""
    host: str = os.getenv("AXUM_CAMERA_HOST", "0.0.0.0")
    port: int = _env_int("AXUM_CAMERA_PORT", 5001)
    capture_width: int | None = _env_optional_int("AXUM_CAPTURE_WIDTH")
    capture_height: int | None = _env_optional_int("AXUM_CAPTURE_HEIGHT")
    capture_quality: int = _env_int("AXUM_CAPTURE_JPEG_QUALITY", 96)
    stream_width: int = _env_int("AXUM_STREAM_WIDTH", 640)
    stream_height: int = _env_int("AXUM_STREAM_HEIGHT", 480)
    stream_fps: int = _env_int("AXUM_STREAM_FPS", 10)
    stream_quality: int = _env_int("AXUM_STREAM_JPEG_QUALITY", 75)
    warmup_seconds: int = _env_int("AXUM_CAMERA_WARMUP_SECONDS", 2, minimum=0)
    stale_after_seconds: int = _env_int("AXUM_STREAM_STALE_SECONDS", 5)

    def __post_init__(self) -> None:
        for name in ("capture_quality", "stream_quality"):
            value = getattr(self, name)
            if value > 100:
                raise ValueError(f"{name} must be between 1 and 100")
        if (self.capture_width is None) != (self.capture_height is None):
            raise ValueError("AXUM_CAPTURE_WIDTH and AXUM_CAPTURE_HEIGHT must be set together")

class CameraService:
    """Own one Picamera2 instance configured for simultaneous still and preview streams."""

    def __init__(
        self,
        settings: CameraSettings | None = None,
        camera_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.settings = settings or CameraSettings()
        self._camera_factory = camera_factory
        self._camera: Any | None = None
        self._camera_lock = threading.Lock()
        self._frame_condition = threading.Condition()
        self._latest_stream_frame = b""
        self._last_frame_at: float | None = None
        self._started_at = time.monotonic()
        self._capture_count = 0
        self._last_error: str | None = None
        self._stop_event = threading.Event()
        self._stream_thread: threading.Thread | None = None
        self.capture_size: tuple[int, int] | None = None
        self.sensor_resolution: tuple[int, int] | None = None
        self.camera_model: str | None = None

    @property
    def ready(self) -> bool:
        return self._camera is not None and self._last_error is None

    def start(self) -> None:
        if self._camera is not None:
            return
        try:
            factory = self._camera_factory
            if factory is None:
                if not PICAMERA_AVAILABLE or Picamera2 is None:
                    raise RuntimeError("picamera2 is unavailable; install python3-picamera2 on Raspberry Pi OS")
                factory = Picamera2
            camera = factory()
            sensor_resolution = tuple(int(v) for v in camera.sensor_resolution)
            capture_size = (
                (self.settings.capture_width, self.settings.capture_height)
                if self.settings.capture_width is not None
                else sensor_resolution
            )
            assert capture_size[0] is not None and capture_size[1] is not None

            config = camera.create_video_configuration(
                main={"size": capture_size, "format": "BGR888"},
                lores={
                    "size": (self.settings.stream_width, self.settings.stream_height),
                    "format": "BGR888",
                },
                controls={"FrameRate": self.settings.stream_fps},
                buffer_count=4,
            )
            camera.configure(config)
            camera.start()
            if self.settings.warmup_seconds:
                time.sleep(self.settings.warmup_seconds)

            properties = getattr(camera, "camera_properties", {}) or {}
            self._camera = camera
            self.capture_size = (int(capture_size[0]), int(capture_size[1]))
            self.sensor_resolution = sensor_resolution
            self.camera_model = str(properties.get("Model") or properties.get("CameraModel") or "unknown")
            self._last_error = None
            self._stop_event.clear()
            self._stream_thread = threading.Thread(
                target=self._stream_worker,
                name="axum-camera-preview",
                daemon=True,
            )
            self._stream_thread.start()
            LOGGER.info(
                "Camera ready model=%s still=%sx%s preview=%sx%s@%sfps",
                self.camera_model,
                *self.capture_size,
                self.settings.stream_width,
                self.settings.stream_height,
                self.settings.stream_fps,
            )
        except Exception as exc:
            self._last_error = str(exc)
            LOGGER.exception("Camera startup failed")
            self.stop()
            raise

    def stop(self) -> None:
        self._stop_event.set()
        with self._frame_condition:
            self._frame_condition.notify_all()
        thread = self._stream_thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._stream_thread = None
        camera, self._camera = self._camera, None
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                LOGGER.exception("Camera stop failed")
            try:
                camera.close()
            except Exception:
                LOGGER.exception("Camera close failed")

    def _encode(self, frame: Any, quality: int) -> bytes:
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("OpenCV failed to encode camera frame as JPEG")
        return encoded.tobytes()

    def capture_jpeg(self) -> bytes:
        if self._camera is None:
            raise RuntimeError(self._last_error or "camera is not ready")
        with self._camera_lock:
            frame = self._camera.capture_array("main")
        jpeg = self._encode(frame, self.settings.capture_quality)
        self._capture_count += 1
        return jpeg

    def _stream_worker(self) -> None:
        interval = 1.0 / self.settings.stream_fps
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                if self._camera is None:
                    return
                with self._camera_lock:
                    frame = self._camera.capture_array("lores")
                jpeg = self._encode(frame, self.settings.stream_quality)
                with self._frame_condition:
                    self._latest_stream_frame = jpeg
                    self._last_frame_at = time.monotonic()
                    self._last_error = None
                    self._frame_condition.notify_all()
            except Exception as exc:
                self._last_error = str(exc)
                LOGGER.exception("Preview capture failed")
                self._stop_event.wait(min(interval, 1.0))
                continue
            remaining = interval - (time.monotonic() - started)
            if remaining > 0:
                self._stop_event.wait(remaining)

    def wait_for_stream_frame(self, previous: bytes | None = None, timeout: float = 2.0) -> bytes:
        deadline = time.monotonic() + timeout
        with self._frame_condition:
            while not self._stop_event.is_set():
                frame = self._latest_stream_frame
                if frame and frame is not previous:
                    return frame
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._frame_condition.wait(remaining)
        raise TimeoutError("preview frame unavailable")

    def status_payload(self) -> dict[str, Any]:
        now = time.monotonic()
        frame_age = None if self._last_frame_at is None else round(now - self._last_frame_at, 3)
        stream_alive = bool(self._stream_thread and self._stream_thread.is_alive())
        healthy = self.ready and stream_alive and frame_age is not None and frame_age <= self.settings.stale_after_seconds
        return {
            "ok": healthy,
            "camera_ready": self._camera is not None,
            "camera_model": self.camera_model,
            "sensor_resolution": list(self.sensor_resolution) if self.sensor_resolution else None,
            "capture_resolution": list(self.capture_size) if self.capture_size else None,
            "capture_jpeg_quality": self.settings.capture_quality,
            "stream_resolution": [self.settings.stream_width, self.settings.stream_height],
            "stream_fps": self.settings.stream_fps,
            "stream_alive": stream_alive,
            "last_stream_frame_age_seconds": frame_age,
            "capture_count": self._capture_count,
            "uptime_seconds": round(now - self._started_at, 1),
            "ir_control": "automatic_hardware",
            "last_error": self._last_error,
        }


def _no_cache(response: Response) -> Response:
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


def create_app(camera_service: CameraService | None = None) -> Flask:
    app = Flask(__name__)
    service = camera_service or CameraService()
    app.extensions["axum_camera_service"] = service

    @app.get("/capture")
    def capture() -> Response:
        try:
            return _no_cache(Response(service.capture_jpeg(), mimetype="image/jpeg"))
        except Exception as exc:
            LOGGER.exception("Still capture failed")
            return jsonify({"ok": False, "error": str(exc)}), 503

    def mjpeg_generator():
        previous: bytes | None = None
        while True:
            try:
                frame = service.wait_for_stream_frame(previous)
                previous = frame
            except TimeoutError:
                if not service.ready:
                    return
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

    @app.get("/stream")
    def stream() -> Response:
        if not service.ready:
            return jsonify({"ok": False, "error": service.status_payload()["last_error"]}), 503
        return _no_cache(
            Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")
        )

    @app.get("/status")
    def status() -> Response:
        payload = service.status_payload()
        return jsonify(payload), 200 if payload["ok"] else 503

    return app


SETTINGS = CameraSettings()
CAMERA_SERVICE = CameraService(SETTINGS)
app = create_app(CAMERA_SERVICE)


def _shutdown() -> None:
    CAMERA_SERVICE.stop()


def _handle_signal(_signum: int, _frame: Any) -> None:
    CAMERA_SERVICE.stop()
    raise SystemExit(0)


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("AXUM_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    atexit.register(_shutdown)
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    try:
        CAMERA_SERVICE.start()
    except Exception:
        LOGGER.error("Starting HTTP service in degraded mode; /status will return 503")
    app.run(host=SETTINGS.host, port=SETTINGS.port, threaded=True, use_reloader=False)
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