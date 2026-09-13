from __future__ import annotations

from pathlib import Path

import requests

from src.arm.controller import CameraInterface, TurntableController


class FakeResponse:
    def __init__(self, *, content=b"", payload=None, content_type="application/json", error=None):
        self.content = content
        self._payload = payload
        self.headers = {"content-type": content_type}
        self._error = error

    def raise_for_status(self) -> None:
        if self._error:
            raise self._error

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses, events=None, post_responses=None):
        self.responses = list(responses)
        self.post_responses = list(post_responses or [])
        self.events = events if events is not None else []
        self.urls = []

    def get(self, url, timeout):
        self.urls.append(url)
        self.events.append(("http", url))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def post(self, url, json=None, timeout=None):
        self.events.append(("post", url, json["locked"]))
        if not self.post_responses:
            raise requests.ConnectionError("no such endpoint")
        response = self.post_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_camera_interface_derives_pi_endpoints_and_status() -> None:
    session = FakeSession([FakeResponse(payload={"ok": True, "camera_model": "OV5647"})])
    camera = CameraInterface("http://pi.local:5001/stream", session=session)
    payload = camera.status()
    assert camera.capture_url == "http://pi.local:5001/capture"
    assert camera.status_url == "http://pi.local:5001/status"
    assert session.urls == [camera.status_url]
    assert payload["camera_model"] == "OV5647"


def test_capture_retries_and_preserves_jpeg_bytes(tmp_path: Path, monkeypatch) -> None:
    jpeg = b"\xff\xd8raw-jpeg-bytes\xff\xd9"
    session = FakeSession([
        requests.ConnectionError("wifi drop"),
        FakeResponse(content=jpeg, content_type="image/jpeg"),
    ])
    monkeypatch.setattr("src.arm.controller.time.sleep", lambda _seconds: None)
    camera = CameraInterface("http://pi.local:5001/stream", retries=2, session=session)
    output = tmp_path / "capture.jpg"
    result = camera.capture_frame(output)
    assert result == jpeg
    assert output.read_bytes() == jpeg
    assert session.urls == [camera.capture_url, camera.capture_url]


def test_turntable_preflights_pi_before_photo_command(tmp_path: Path) -> None:
    events = []
    jpeg = b"\xff\xd8frame\xff\xd9"
    session = FakeSession(
        [
            FakeResponse(payload={"ok": True}),
            FakeResponse(content=jpeg, content_type="image/jpeg"),
        ],
        events,
        post_responses=[
            FakeResponse(payload={"ok": True, "controls": {"AeEnable": False}}),
            FakeResponse(payload={"ok": True, "controls": {"AeEnable": True}}),
        ],
    )
    camera = CameraInterface(
        "http://pi.local:5001/stream",
        require_status=True,
        session=session,
    )

    class FakeArduino:
        def send_command(self, command, expect_prefix=None):
            events.append(("arduino", command))
            return f"OK:{command.split(':', 1)[0]}"

    turntable = TurntableController(arduino=FakeArduino(), camera=camera)
    frames = turntable.capture_rotation_set("TEST", photo_count=1, output_dir=tmp_path)
    assert len(frames) == 1
    assert (tmp_path / "TEST_000.jpg").read_bytes() == jpeg
    assert events == [
        ("http", camera.status_url),
        ("post", camera.lock_url, True),
        ("arduino", "PHOTO"),
        ("http", camera.capture_url),
        ("post", camera.lock_url, False),
    ]


def test_turntable_captures_when_the_camera_cannot_lock(tmp_path: Path) -> None:
    """The ESP32-CAM has no /lock, and a drifting set still beats no set."""
    events = []
    jpeg = b"\xff\xd8frame\xff\xd9"
    session = FakeSession([FakeResponse(content=jpeg, content_type="image/jpeg")], events)
    camera = CameraInterface("http://esp32.local:81/stream", session=session)

    class FakeArduino:
        def send_command(self, command, expect_prefix=None):
            events.append(("arduino", command))
            return f"OK:{command.split(':', 1)[0]}"

    turntable = TurntableController(arduino=FakeArduino(), camera=camera)
    frames = turntable.capture_rotation_set("TEST", photo_count=1, output_dir=tmp_path)
    assert len(frames) == 1
    assert (tmp_path / "TEST_000.jpg").read_bytes() == jpeg
    # It tried once and did not then release a lock it never took.
    assert [event for event in events if event[0] == "post"] == [
        ("post", camera.lock_url, True),
    ]
