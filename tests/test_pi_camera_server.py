from __future__ import annotations

import io
import time
import threading

import cv2
import numpy as np
import pytest
from PIL import Image

from pi.services.pi_camera_server import CameraService, CameraSettings, create_app
from pi.verify_pi import verify_service
from werkzeug.serving import make_server


class FakeCamera:
    sensor_resolution = (320, 240)
    # 320 px * 11340 nm = 3.6288 mm across, the OV5647's real sensor width, so
    # the 35mm-equivalent focal length works out to a checkable round number.
    camera_properties = {
        "Model": "AXUM-FakeCam",
        "UnitCellSize": (11340, 11340),
        "PixelArraySize": (320, 240),
    }
    metadata = {"ExposureTime": 19998, "AnalogueGain": 2.5, "ColourGains": (1.8, 1.4)}

    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.configuration = None
        self.controls: dict = {}

    def create_video_configuration(self, **kwargs):
        self.configuration = kwargs
        return kwargs

    def configure(self, configuration) -> None:
        self.configuration = configuration

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True

    def set_controls(self, controls) -> None:
        self.controls.update(controls)

    def capture_metadata(self) -> dict:
        return dict(self.metadata)

    def capture_array(self, name: str):
        assert self.started
        if name == "main":
            return np.full((240, 320, 3), (20, 80, 160), dtype=np.uint8)
        if name == "lores":
            return np.full((48, 64, 3), (120, 60, 20), dtype=np.uint8)
        raise KeyError(name)


class MeterlessCamera(FakeCamera):
    """A camera that reports no exposure -- there is nothing to pin."""
    metadata: dict = {}


def make_service(focal_mm: float | None = None, factory=FakeCamera) -> CameraService:
    settings = CameraSettings(
        capture_width=None,
        capture_height=None,
        lens_focal_mm=focal_mm,
        stream_width=64,
        stream_height=48,
        stream_fps=30,
        warmup_seconds=0,
        stale_after_seconds=2,
    )
    service = CameraService(settings, camera_factory=factory)
    service.start()
    deadline = time.monotonic() + 2
    while not service.status_payload()["ok"] and time.monotonic() < deadline:
        time.sleep(0.01)
    return service


def decode(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return image


def test_status_reports_dual_stream_configuration() -> None:
    service = make_service()
    try:
        client = create_app(service).test_client()
        response = client.get("/status")
        payload = response.get_json()
        assert response.status_code == 200
        assert payload["ok"] is True
        assert payload["camera_model"] == "AXUM-FakeCam"
        assert payload["sensor_resolution"] == [320, 240]
        assert payload["capture_resolution"] == [320, 240]
        assert payload["stream_resolution"] == [64, 48]
        assert payload["ir_control"] == "automatic_hardware"
    finally:
        service.stop()


def test_capture_returns_full_resolution_jpeg_without_cache() -> None:
    service = make_service()
    try:
        client = create_app(service).test_client()
        response = client.get("/capture")
        image = decode(response.data)
        assert response.status_code == 200
        assert response.content_type == "image/jpeg"
        assert response.headers["Cache-Control"].startswith("no-store")
        assert image.shape[:2] == (240, 320)
        assert service.status_payload()["capture_count"] == 1
    finally:
        service.stop()


def test_stream_yields_low_resolution_mjpeg_frame() -> None:
    service = make_service()
    try:
        client = create_app(service).test_client()
        response = client.get("/stream", buffered=False)
        first_chunk = next(response.response)
        start = first_chunk.index(b"\xff\xd8")
        end = first_chunk.index(b"\xff\xd9", start) + 2
        image = decode(first_chunk[start:end])
        assert response.status_code == 200
        assert "multipart/x-mixed-replace" in response.content_type
        assert image.shape[:2] == (48, 64)
    finally:
        service.stop()


def test_unavailable_camera_returns_503() -> None:
    service = CameraService(CameraSettings(warmup_seconds=0))
    client = create_app(service).test_client()
    status_response = client.get("/status")
    capture_response = client.get("/capture")
    stream_response = client.get("/stream")
    assert status_response.status_code == 503
    assert capture_response.status_code == 503
    assert stream_response.status_code == 503


def test_capture_carries_lens_optics_as_exif() -> None:
    service = make_service(focal_mm=3.6)
    try:
        assert service.sensor_width_mm == pytest.approx(3.6288)
        response = create_app(service).test_client().get("/capture")
        exif = Image.open(io.BytesIO(response.data)).getexif()
        optics = exif.get_ifd(0x8769)
        assert exif[0x0110] == "AXUM-FakeCam"
        assert optics[0x920A] == pytest.approx(3.6)
        # 3.6mm on a 3.6288mm-wide sensor frames what 36mm does on full frame.
        assert optics[0xA405] == 36
    finally:
        service.stop()


def test_capture_omits_exif_when_lens_is_unknown() -> None:
    service = make_service(focal_mm=None)
    try:
        response = create_app(service).test_client().get("/capture")
        assert service.status_payload()["exif_optics"] is False
        assert not Image.open(io.BytesIO(response.data)).getexif()
    finally:
        service.stop()


def test_capture_preserves_channel_order() -> None:
    """picamera2 names formats by memory layout, so a swap here is easy to miss."""
    for focal_mm in (None, 3.6):
        service = make_service(focal_mm=focal_mm)
        try:
            image = decode(create_app(service).test_client().get("/capture").data)
            # int, not uint8: approx subtracts, and 20 - 21 wraps to 255.
            pixel = [int(channel) for channel in image[0, 0]]
            # Tolerance is JPEG's, not the channel order's -- a swap is 140 off.
            assert pixel == pytest.approx([20, 80, 160], abs=3)
        finally:
            service.stop()


def test_lock_pins_the_metered_exposure_and_white_balance() -> None:
    service = make_service()
    try:
        client = create_app(service).test_client()
        response = client.post("/lock", json={"locked": True})
        applied = response.get_json()["controls"]
        assert response.status_code == 200
        assert applied["AeEnable"] is False and applied["AwbEnable"] is False
        assert applied["ExposureTime"] == 19998
        assert applied["AnalogueGain"] == 2.5
        assert applied["ColourGains"] == [1.8, 1.4]
        assert service.status_payload()["capture_locked"] is True

        released = client.post("/lock", json={"locked": False})
        assert released.status_code == 200
        assert released.get_json()["controls"] == {"AeEnable": True, "AwbEnable": True}
        assert service.status_payload()["capture_locked"] is False
    finally:
        service.stop()


def test_lock_refuses_rather_than_disabling_auto_without_a_value() -> None:
    service = make_service(factory=MeterlessCamera)
    try:
        response = create_app(service).test_client().post("/lock", json={"locked": True})
        assert response.status_code == 503
        assert "ExposureTime" in response.get_json()["error"]
        assert service.status_payload()["capture_locked"] is False
    finally:
        service.stop()


def test_lock_requires_an_explicit_state() -> None:
    service = make_service()
    try:
        response = create_app(service).test_client().post("/lock", json={})
        assert response.status_code == 400
    finally:
        service.stop()


def test_verifier_passes_against_live_fake_service() -> None:
    service = make_service(focal_mm=3.6)
    server = make_server("127.0.0.1", 0, create_app(service), threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        results = verify_service(
            f"http://127.0.0.1:{server.server_port}",
            captures=2,
            timeout=3,
        )
        assert all(result.ok for result in results)
        assert {"Concurrent stream and still", "EXIF optics",
                "Exposure and white balance lock"} <= {r.name for r in results}
        # Verifying must not leave the rig pinned to the bench lighting.
        assert service.status_payload()["capture_locked"] is False
    finally:
        server.shutdown()
        thread.join(timeout=2)
        service.stop()
