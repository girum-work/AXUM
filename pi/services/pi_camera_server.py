#!/usr/bin/env python3
"""AXUM Raspberry Pi camera capture and preview service."""

from __future__ import annotations

import atexit
import io
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import cv2
from flask import Flask, Response, jsonify, request
from PIL import Image

try:
    from picamera2 import Picamera2

    PICAMERA_AVAILABLE = True
except ImportError:
    Picamera2 = None  # type: ignore[assignment]
    PICAMERA_AVAILABLE = False

LOGGER = logging.getLogger("axum.pi.camera")

# EXIF tags AliceVision reads when deciding a starting focal length.
EXIF_MAKE = 0x010F
EXIF_MODEL = 0x0110
EXIF_IFD = 0x8769
EXIF_FOCAL_LENGTH = 0x920A
EXIF_FOCAL_LENGTH_35MM = 0xA405
EXIF_PIXEL_X_DIMENSION = 0xA002
EXIF_PIXEL_Y_DIMENSION = 0xA003
FULL_FRAME_WIDTH_MM = 36.0

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

def _env_optional_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw in (None, "", "auto", "AUTO"):
        return None
    value = float(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value

@dataclass(frozen=True)
class CameraSettings:
    """Runtime settings supplied through the systemd environment file."""
    host: str = os.getenv("AXUM_CAMERA_HOST", "0.0.0.0")
    port: int = _env_int("AXUM_CAMERA_PORT", 5001)
    capture_width: int | None = _env_optional_int("AXUM_CAPTURE_WIDTH")
    capture_height: int | None = _env_optional_int("AXUM_CAPTURE_HEIGHT")
    capture_quality: int = _env_int("AXUM_CAPTURE_JPEG_QUALITY", 96)
    # No default: the lens is not something the sensor can report, and an
    # invented focal length is worse than none because photogrammetry would
    # then trust it instead of solving for it.
    lens_focal_mm: float | None = _env_optional_float("AXUM_LENS_FOCAL_MM")
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

def _sensor_width_mm(properties: dict[str, Any]) -> float | None:
    """
    Physical width of the pixel array, from what libcamera reports.

    Measured rather than configured: UnitCellSize is in nanometres per pixel,
    so the sensor's own numbers give the millimetres photogrammetry needs to
    turn a lens focal length into pixels.
    """
    cell = properties.get("UnitCellSize")
    array = properties.get("PixelArraySize")
    if not cell or not array:
        return None
    try:
        return float(cell[0]) * float(array[0]) / 1_000_000.0
    except (TypeError, ValueError, IndexError):
        return None

def _exif_bytes(model: str | None, size: tuple[int, int],
                focal_mm: float | None, sensor_width_mm: float | None) -> bytes | None:
    """
    EXIF block carrying the optics, or None when the lens is unknown.

    Without a focal length AliceVision falls back to a default field of view
    and silently reconstructs at the wrong scale; with it, every image also
    groups into one intrinsic instead of one per file.
    """
    if focal_mm is None:
        return None
    exif = Image.Exif()
    exif[EXIF_MAKE] = "RaspberryPi"
    exif[EXIF_MODEL] = model or "unknown"
    sub: dict[int, Any] = {
        EXIF_FOCAL_LENGTH: focal_mm,
        EXIF_PIXEL_X_DIMENSION: int(size[0]),
        EXIF_PIXEL_Y_DIMENSION: int(size[1]),
    }
    # The 35mm equivalent is the fallback AliceVision uses when its sensor
    # database has no entry for the camera -- and it never has one for these.
    if sensor_width_mm:
        sub[EXIF_FOCAL_LENGTH_35MM] = round(focal_mm * FULL_FRAME_WIDTH_MM / sensor_width_mm)
    exif[EXIF_IFD] = sub
    return exif.tobytes()

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
        self.sensor_width_mm: float | None = None
        self.camera_model: str | None = None
        self._locked_controls: dict[str, Any] | None = None

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

            # "RGB888" yields numpy arrays in BGR order -- picamera2 names
            # formats by memory layout, the reverse of the channel order you
            # index. BGR is what OpenCV and the rest of this codebase expect;
            # "BGR888" here would swap red and blue in every frame.
            config = camera.create_video_configuration(
                main={"size": capture_size, "format": "RGB888"},
                lores={
                    "size": (self.settings.stream_width, self.settings.stream_height),
                    "format": "RGB888",
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
            self.sensor_width_mm = _sensor_width_mm(properties)
            self.camera_model = str(properties.get("Model") or properties.get("CameraModel") or "unknown")
            self._locked_controls = None
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

    def _encode_still(self, frame: Any) -> bytes:
        """Encode a capture through Pillow, which can carry EXIF; cv2 cannot."""
        height, width = frame.shape[:2]
        exif = _exif_bytes(self.camera_model, (width, height),
                           self.settings.lens_focal_mm, self.sensor_width_mm)
        if exif is None:
            return self._encode(frame, self.settings.capture_quality)
        buffer = io.BytesIO()
        Image.fromarray(frame[:, :, ::-1]).save(
            buffer, format="JPEG", quality=self.settings.capture_quality, exif=exif)
        return buffer.getvalue()

    def capture_jpeg(self) -> bytes:
        if self._camera is None:
            raise RuntimeError(self._last_error or "camera is not ready")
        with self._camera_lock:
            frame = self._camera.capture_array("main")
        jpeg = self._encode_still(frame)
        self._capture_count += 1
        return jpeg

    def set_capture_lock(self, locked: bool) -> dict[str, Any]:
        """
        Freeze or release exposure and white balance.

        A turntable set is one scene photographed 36 times. Left on auto, the
        camera re-meters as the object turns, so brightness and colour drift
        across the set -- which weakens feature matching and gives a blotchy
        texture. Locking samples the state the camera has already converged on
        (the preview thread has been metering continuously) and pins it.

        Returns:
            The controls applied, for the caller to log or verify.
        """
        if self._camera is None:
            raise RuntimeError(self._last_error or "camera is not ready")
        if not locked:
            controls: dict[str, Any] = {"AeEnable": True, "AwbEnable": True}
            with self._camera_lock:
                self._camera.set_controls(controls)
            self._locked_controls = None
            LOGGER.info("Exposure and white balance released to auto")
            return controls

        with self._camera_lock:
            metadata = self._camera.capture_metadata() or {}
        controls = {"AeEnable": False, "AwbEnable": False}
        for key in ("ExposureTime", "AnalogueGain", "ColourGains"):
            if metadata.get(key) is not None:
                controls[key] = metadata[key]
        missing = [k for k in ("ExposureTime", "AnalogueGain", "ColourGains")
                   if k not in controls]
        if missing:
            # Disabling auto without pinning the value leaves the camera on
            # whatever it last chose, which is not a lock anyone can rely on.
            raise RuntimeError(
                f"Camera did not report {', '.join(missing)}; cannot lock exposure")
        with self._camera_lock:
            self._camera.set_controls(controls)
        self._locked_controls = controls
        LOGGER.info("Exposure and white balance locked: %s", controls)
        return controls

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
            "sensor_width_mm": self.sensor_width_mm,
            "lens_focal_mm": self.settings.lens_focal_mm,
            "exif_optics": self.settings.lens_focal_mm is not None,
            "capture_locked": self._locked_controls is not None,
            "locked_controls": self._locked_controls,
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

    @app.post("/lock")
    def lock() -> Response:
        body = request.get_json(silent=True) or {}
        if "locked" not in body:
            return jsonify({"ok": False, "error": "body must contain 'locked'"}), 400
        try:
            controls = service.set_capture_lock(bool(body["locked"]))
        except Exception as exc:
            LOGGER.exception("Exposure lock failed")
            return jsonify({"ok": False, "error": str(exc)}), 503
        return jsonify({"ok": True, "locked": bool(body["locked"]), "controls": controls})

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