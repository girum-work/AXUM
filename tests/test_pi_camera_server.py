from __future__ import annotations

import time
import threading

import cv2
import numpy as np

from pi.services.pi_camera_server import CameraService, CameraSettings, create_app
from pi.verify_pi import verify_service
from werkzeug.serving import make_server


class FakeCamera:
    sensor_resolution = (320, 240)
    camera_properties = {"Model": "AXUM-FakeCam"}

    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.configuration = None

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

    def capture_array(self, name: str):
        assert self.started
        if name == "main":
            return np.full((240, 320, 3), (20, 80, 160), dtype=np.uint8)
        if name == "lores":
            return np.full((48, 64, 3), (120, 60, 20), dtype=np.uint8)
        raise KeyError(name)


def make_service() -> CameraService:
    settings = CameraSettings(
        capture_width=None,
        capture_height=None,
        stream_width=64,
        stream_height=48,
        stream_fps=30,
        warmup_seconds=0,
        stale_after_seconds=2,
    )
    service = CameraService(settings, camera_factory=FakeCamera)
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


def test_verifier_passes_against_live_fake_service() -> None:
    service = make_service()
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
        assert any(result.name == "Concurrent stream and still" for result in results)
    finally:
        server.shutdown()
        thread.join(timeout=2)
        service.stop()
