"""
AXUM ROVER - Arduino and camera control layer.

WHAT: Safe wrappers for Arduino serial commands, arm poses, turntable scans,
and ESP32-CAM still-image capture.
WHY: The rest of the project should never touch ``serial.write`` directly;
all hardware traffic is centralised here so failures can degrade cleanly.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import serial
import threading
from loguru import logger
from serial.tools import list_ports

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    ESP32_CAM_URL,
    SCAN_PHOTOS_DIR,
    SERIAL_BAUD,
    SERIAL_PORT,
    TURNTABLE_SETTLE_MS,
    TURNTABLE_STEPS,
)


class ArduinoSerial:
    """
    Serial command gateway for the Arduino Mega.

    Args:
        port: Optional explicit COM port. If omitted, config.SERIAL_PORT is
            tried first, then connected serial ports are probed with PING.
        baudrate: Serial baud rate used by the firmware.
        timeout: Read timeout in seconds for each command response.
        retries: Number of command attempts before raising an error.
        connect: When False, defer opening the serial connection.
    """

    def __init__(
        self,
        port: str | None = None,
        baudrate: int = SERIAL_BAUD,
        timeout: float = 2.0,
        retries: int = 2,
        connect: bool = True,
    ) -> None:
        self.port = port or SERIAL_PORT
        self.baudrate = baudrate
        self.timeout = timeout
        self.retries = retries
        self.serial: serial.Serial | None = None
        # Guards the write/read transaction below. Needed because
        # DriveController's watchdog thread (below) calls send_command()
        # from a background thread independently of whatever thread issued
        # the actual drive/stop command -- without this lock, two threads
        # writing to the same serial connection concurrently can interleave
        # bytes or read each other's response. Merged in from a
        # collected copy of this file (Robotics Software Engineer) that
        # already had DriveController + this lock; this integration pass's
        # own version had neither.
        self._io_lock = threading.Lock()
        if connect:
            self.connect()

    def connect(self) -> None:
        """
        Open a serial connection to the Arduino.

        Raises:
            ConnectionError: If no configured or auto-detected port responds.
        """
        port = self.port if self.port else self._find_arduino_port()
        if not port:
            raise ConnectionError("Arduino serial port not found")

        try:
            self.serial = serial.Serial(port, self.baudrate, timeout=self.timeout)
            time.sleep(2.0)
            self._drain_startup_lines()
            self.port = port
            logger.info(f"Arduino connected on {port}")
        except serial.SerialException as exc:
            self.serial = None
            raise ConnectionError(f"Could not open Arduino port {port}: {exc}") from exc

    def close(self) -> None:
        """Close the serial connection if it is open."""
        if self.serial and self.serial.is_open:
            self.serial.close()
            logger.info("Arduino serial connection closed")

    def _drain_startup_lines(self) -> None:
        """Read firmware startup banner lines so the next command is clean."""
        if not self.serial:
            return
        deadline = time.time() + 0.5
        while time.time() < deadline and self.serial.in_waiting:
            self.serial.readline()

    def _find_arduino_port(self) -> str | None:
        """
        Probe available serial ports and return the first Arduino-like port.

        The method prefers config.SERIAL_PORT when it exists. If that is not
        available, USB serial devices with Arduino-ish descriptions are tried.
        """
        ports = list(list_ports.comports())
        known = {p.device for p in ports}
        if self.port in known:
            return self.port

        preferred = [
            p.device for p in ports
            if any(token in (p.description or "").lower()
                   for token in ("arduino", "mega", "ch340", "usb serial", "usb-serial"))
        ]
        candidates = preferred or [p.device for p in ports]
        for candidate in candidates:
            try:
                with serial.Serial(candidate, self.baudrate, timeout=1.0) as probe:
                    time.sleep(2.0)
                    while probe.in_waiting:
                        probe.readline()
                    probe.write(b"PING\n")
                    response = probe.readline().decode("utf-8", errors="replace").strip()
                    if response == "PONG":
                        return candidate
            except serial.SerialException:
                continue
        return None

    def send_command(self, command: str, expect_prefix: str | None = None) -> str:
        """
        Send one newline-terminated command and return the Arduino response.

        Args:
            command: ASCII command without trailing newline, e.g. ``PING``.
            expect_prefix: Optional accepted response prefix such as ``OK:``.

        Raises:
            RuntimeError: If the command fails or gets an unexpected response.
        """
        if self.serial is None or not self.serial.is_open:
            self.connect()

        assert self.serial is not None
        payload = f"{command.strip()}\n".encode("ascii", errors="strict")
        last_error = ""

        for attempt in range(self.retries + 1):
            try:
                with self._io_lock:
                    self.serial.reset_input_buffer()
                    self.serial.write(payload)
                    self.serial.flush()
                    response = self.serial.readline().decode("utf-8", errors="replace").strip()
                if not response:
                    last_error = "empty response"
                    continue
                if response.startswith("ERR:"):
                    raise RuntimeError(f"Arduino rejected {command!r}: {response}")
                if expect_prefix and not response.startswith(expect_prefix):
                    last_error = f"unexpected response {response!r}"
                    continue
                logger.debug(f"Arduino {command!r} -> {response!r}")
                return response
            except (serial.SerialException, OSError) as exc:
                last_error = str(exc)
                logger.warning(f"Serial attempt {attempt + 1} failed: {exc}")
                self.close()
                time.sleep(0.2)
                if attempt < self.retries:
                    self.connect()

        raise RuntimeError(f"Arduino command {command!r} failed: {last_error}")

    def ping(self) -> bool:
        """Return True when the Arduino responds to PING with PONG."""
        try:
            return self.send_command("PING") == "PONG"
        except Exception as exc:
            logger.warning(f"Arduino ping failed: {exc}")
            return False

    def status(self) -> dict[str, Any]:
        """Return parsed STATUS JSON from the Arduino, or an empty dict."""
        response = self.send_command("STATUS")
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse Arduino STATUS: {response}")
            return {}

    def gps_status(self) -> "GPSFix":
        """
        Query the Arduino's GPS_STATUS command and return a parsed fix.

        Format: JSON, matching STATUS's pattern -- {"fix": bool, "lat":
        float, "lon": float, "sats": int}. Outdoor/transport-phase
        navigation only -- indoor arena navigation still uses ArUco via
        the Pi, unrelated to this.

        GPS is polled on request, not pushed continuously by firmware --
        callers are responsible for calling this at whatever cadence they
        need fresh data. NEO-7M modules update at 1Hz by default; polling
        faster than that doesn't get fresher data, just adds serial
        traffic for no benefit.

        Returns:
            GPSFix with has_fix=False and lat/lng/satellites=None if no
            fix is available or the response doesn't parse, or a
            populated GPSFix if a fix exists. Never raises -- a malformed
            GPS response shouldn't crash a caller any more than a
            malformed STATUS response would.
        """
        response = self.send_command("GPS_STATUS")
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse Arduino GPS_STATUS: {response!r}")
            return GPSFix(has_fix=False)

        if not payload.get("fix"):
            return GPSFix(has_fix=False)

        try:
            return GPSFix(
                has_fix=True,
                latitude=float(payload["lat"]),
                longitude=float(payload["lon"]),
                satellites=int(payload["sats"]),
            )
        except (KeyError, TypeError, ValueError):
            logger.warning(f"Could not parse Arduino GPS_STATUS fields: {response!r}")
            return GPSFix(has_fix=False)

    def cam_arm_trigger(self) -> str:
        """
        Trigger the arm-angle ESP32-CAM via its Serial2 UART link to the
        Mega. This only sends the trigger -- the resulting JPEG still needs
        to be fetched separately over WiFi HTTP (CameraInterface), same as
        the undercarriage camera. See axum_rover.ino integration note #6.
        """
        return self.send_command("CAM_ARM_TRIGGER", expect_prefix="OK:")


@dataclass
class GPSFix:
    """
    Parsed result of a GPS_STATUS query.

    Args:
        has_fix: Whether the module currently reports a valid position.
        latitude: Decimal degrees, None if has_fix is False.
        longitude: Decimal degrees, None if has_fix is False.
        satellites: Satellite count contributing to the fix, None if
            has_fix is False.
    """

    has_fix: bool
    latitude: float | None = None
    longitude: float | None = None
    satellites: int | None = None


class ArmController:
    """
    High-level arm and gripper motions.

    Args:
        arduino: Shared ArduinoSerial instance used for all commands.
    """

    def __init__(self, arduino: ArduinoSerial | None = None) -> None:
        self.arduino = arduino or ArduinoSerial()
        # Tracks the last commanded joint angles so callers can query
        # current arm state without a round-trip to firmware. For AI & CV
        # Engineer's hand-eye calibration (AX=XB solve), which needs each
        # calibration image paired with the arm's pose at capture time.
        # NOTE: this is the last COMMANDED state, not a verified read-back
        # from the servos themselves -- firmware doesn't report actual
        # servo position, only accepts commands. If a servo stalls or
        # doesn't reach the commanded angle, this will be wrong and there
        # is currently no way to detect that.
        self._last_angles: tuple[int, int, int, int] | None = None

    def go_pose(self, name: str) -> str:
        """Move the arm to a named firmware pose such as PARK or HOVER_TRAY."""
        return self.arduino.send_command(f"POSE:{name.upper()}", expect_prefix="OK:")

    def set_angles(self, s0: int, s1: int, s2: int, s3: int) -> str:
        """Move the four arm joints to explicit servo angles in degrees."""
        values = [self._clamp_angle(v) for v in (s0, s1, s2, s3)]
        result = self.arduino.send_command(
            f"ARM:{values[0]},{values[1]},{values[2]},{values[3]}",
            expect_prefix="OK:",
        )
        self._last_angles = (values[0], values[1], values[2], values[3])
        return result

    def get_current_angles(self) -> tuple[int, int, int, int] | None:
        """
        Return the last commanded (s0, s1, s2, s3) joint angles, or None
        if set_angles() has never been called on this instance (e.g. only
        go_pose() has been used so far -- named poses don't currently
        report their underlying joint angles back to this layer).

        Returns joint angles, not Cartesian end-effector pose -- there is
        no forward kinematics in this codebase. Poses are a pre-computed
        lookup table on the firmware side, not derived from a kinematic
        model.
        """
        return self._last_angles

    def grip_pull(self) -> str:
        """
        Engage the syringe-actuated vacuum gripper (pull = grip).

        Matches firmware's GRIP:PULL command. The old numeric GRIP:<angle>
        protocol is retired in firmware as of this integration pass — this
        method (not set_grip) is the only supported way to close the
        gripper now.
        """
        return self.arduino.send_command("GRIP:PULL", expect_prefix="OK:")

    def grip_release(self) -> str:
        """Release the syringe-actuated vacuum gripper. Matches GRIP:RELEASE."""
        return self.arduino.send_command("GRIP:RELEASE", expect_prefix="OK:")

    def camera_tilt(self, angle: int) -> str:
        """Set the rear-arm camera tilt servo angle in degrees (0-180)."""
        return self.arduino.send_command(
            f"CAMERA_TILT:{self._clamp_angle(angle)}",
            expect_prefix="OK:",
        )

    def pick_from_tray(self) -> None:
        """Run the standard tray pick sequence used by the demo mission."""
        self.go_pose("HOVER_TRAY")
        self.go_pose("GRIP_TRAY")
        self.grip_pull()
        self.go_pose("LIFT_CLEAR")

    def place_on_turntable(self) -> None:
        """Run the standard placement sequence onto the scan turntable."""
        self.go_pose("HOVER_TABLE")
        self.go_pose("PLACE_TABLE")
        self.grip_release()
        self.go_pose("LIFT_TABLE")

    def park(self) -> None:
        """Move the arm to the safe parked pose."""
        self.go_pose("PARK")

    @staticmethod
    def _clamp_angle(value: int) -> int:
        """Clamp servo angle to the firmware-supported 0-180 range."""
        return max(0, min(180, int(value)))


class DriveController:
    """
    Differential drivetrain abstraction over the Arduino's DRIVE command.

    Args:
        arduino: Shared ArduinoSerial instance.
        watchdog_seconds: Auto-stop if no drive command refreshes within
            this window. Keep this short — it should feel instant to a
            person holding a button, not like a delay.
    """

    def __init__(self, arduino: "ArduinoSerial", watchdog_seconds: float = 0.4) -> None:
        self.arduino = arduino
        self.watchdog_seconds = watchdog_seconds
        self._lock = threading.Lock()
        self._last_command_time = 0.0
        self._stopped = True
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def drive(self, left_speed: int, right_speed: int) -> str:
        """
        Send raw differential speeds. Call repeatedly while a direction is
        held — a single call will be auto-stopped by the watchdog.

        Args:
            left_speed: -255 to 255.
            right_speed: -255 to 255.
        """
        left = self._clamp_speed(left_speed)
        right = self._clamp_speed(right_speed)
        with self._lock:
            self._last_command_time = time.monotonic()
            self._stopped = False
        return self.arduino.send_command(f"DRIVE:{left},{right}", expect_prefix="OK:")

    def forward(self, speed: int = 150) -> str:
        return self.drive(speed, speed)

    def backward(self, speed: int = 150) -> str:
        return self.drive(-speed, -speed)

    def turn_left(self, speed: int = 120) -> str:
        return self.drive(-speed, speed)

    def turn_right(self, speed: int = 120) -> str:
        return self.drive(speed, -speed)

    def stop(self) -> str:
        """Immediate stop. Never blocked by the watchdog lock semantics."""
        with self._lock:
            self._stopped = True
        return self.arduino.send_command("STOP", expect_prefix="OK:")

    def _watchdog_loop(self) -> None:
        """Background thread: auto-stop if a drive command isn't refreshed."""
        while True:
            time.sleep(0.05)
            with self._lock:
                stopped = self._stopped
                elapsed = time.monotonic() - self._last_command_time
            if not stopped and elapsed > self.watchdog_seconds:
                try:
                    self.arduino.send_command("STOP", expect_prefix="OK:")
                    logger.debug("DriveController watchdog: auto-stopped (no refresh)")
                except Exception as exc:
                    logger.warning(f"DriveController watchdog STOP failed: {exc}")
                with self._lock:
                    self._stopped = True

    @staticmethod
    def _clamp_speed(value: int) -> int:
        return max(-255, min(255, int(value)))


@dataclass
class CapturedFrame:
    """
    One captured scan image.

    Attributes:
        path: Location where the JPEG was saved.
        index: Rotation index in the capture set.
        angle_degrees: Approximate turntable angle for the frame.
    """

    path: Path
    index: int
    angle_degrees: float


class CameraInterface:
    """
    HTTP client for ESP32-CAM capture and stream endpoints.

    Args:
        stream_url: Configured stream URL, usually ending in ``:81/stream``.
        timeout: HTTP timeout in seconds.
    """

    def __init__(self, stream_url: str = ESP32_CAM_URL, timeout: float = 8.0) -> None:
        self.stream_url = stream_url
        self.timeout = timeout
        self.capture_url = self._capture_url_from_stream(stream_url)

    def capture_frame(self, output_path: Path | None = None) -> bytes:
        """
        Capture one JPEG from ESP32-CAM and optionally save it to disk.

        Args:
            output_path: Optional path where the JPEG bytes should be written.

        Returns:
            Raw JPEG bytes.
        """
        response = requests.get(self.capture_url, timeout=self.timeout)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "image" not in content_type and not response.content.startswith(b"\xff\xd8"):
            raise RuntimeError(f"ESP32 capture did not return an image: {content_type}")

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(response.content)
        return response.content

    def stream_endpoint(self) -> str:
        """Return the MJPEG stream URL for dashboards or manual checks."""
        return self.stream_url

    @staticmethod
    def _capture_url_from_stream(stream_url: str) -> str:
        """Convert a configured stream URL into the still capture URL."""
        if stream_url.endswith(":81/stream"):
            return stream_url.replace(":81/stream", "/capture")
        if stream_url.endswith("/stream"):
            return stream_url[:-len("/stream")] + "/capture"
        return stream_url.rstrip("/") + "/capture"


class TurntableController:
    """
    High-level turntable motion and photo capture.

    Args:
        arduino: Shared ArduinoSerial instance.
        camera: Optional ESP32 camera client for saving still images.
    """

    def __init__(
        self,
        arduino: ArduinoSerial | None = None,
        camera: CameraInterface | None = None,
    ) -> None:
        self.arduino = arduino or ArduinoSerial()
        self.camera = camera or CameraInterface()

    def step(self, steps: int) -> str:
        """Rotate the turntable by raw firmware step count."""
        return self.arduino.send_command(f"STEP:{int(steps)}", expect_prefix="OK:")

    def rotate_degrees(self, degrees: float) -> str:
        """Rotate the turntable by degrees using firmware conversion."""
        return self.arduino.send_command(f"ROTATE:{float(degrees):.2f}", expect_prefix="OK:")

    def trigger_photo(self) -> str:
        """Pulse the Arduino camera trigger pin."""
        return self.arduino.send_command("PHOTO", expect_prefix="OK:")

    def capture_rotation_set(
        self,
        object_id: str,
        photo_count: int = TURNTABLE_STEPS,
        output_dir: Path | None = None,
    ) -> list[CapturedFrame]:
        """
        Capture a full turntable image set for photogrammetry.

        Args:
            object_id: Catalogue ID used for the output folder name.
            photo_count: Number of evenly-spaced photos over 360 degrees.
            output_dir: Optional directory overriding scans/photos/<object_id>.

        Returns:
            CapturedFrame metadata for every saved JPEG.
        """
        if photo_count <= 0:
            raise ValueError("photo_count must be positive")

        output_dir = Path(output_dir or (SCAN_PHOTOS_DIR / object_id))
        output_dir.mkdir(parents=True, exist_ok=True)
        degrees = 360.0 / photo_count
        frames: list[CapturedFrame] = []

        for index in range(photo_count):
            if index > 0:
                self.rotate_degrees(degrees)
                time.sleep(TURNTABLE_SETTLE_MS / 1000.0)
            self.trigger_photo()
            path = output_dir / f"{object_id}_{index:03d}.jpg"
            self.camera.capture_frame(path)
            frames.append(CapturedFrame(path=path, index=index, angle_degrees=index * degrees))

        return frames


class LightingController:
    """
    Photometric-stereo quadrant LEDs + UV fluorescence LED.

    Args:
        arduino: Shared ArduinoSerial instance.

    Note: matches firmware's LED:QUAD:<N|E|S|W|OFF> and LED:UV:ON/OFF
    commands, added in this integration pass alongside the firmware
    rewrite. Nothing in main_pipeline.py calls this yet — photometric
    stereo / multispectral capture sequences still need to be wired to
    fire the right quadrant/UV LED before each frame, once those analysis
    modules themselves exist (see ER-11).
    """

    def __init__(self, arduino: ArduinoSerial | None = None) -> None:
        self.arduino = arduino or ArduinoSerial()

    def quadrant(self, which: str) -> str:
        """
        Fire one photometric-stereo quadrant LED, all others off.

        Args:
            which: One of "N", "E", "S", "W", or "OFF" to turn all off.
        """
        which = which.upper()
        if which not in {"N", "E", "S", "W", "OFF"}:
            raise ValueError(f"Unknown LED quadrant: {which!r}")
        return self.arduino.send_command(f"LED:QUAD:{which}", expect_prefix="OK:")

    def quadrants_off(self) -> str:
        """Turn all four quadrant LEDs off."""
        return self.quadrant("OFF")

    def uv(self, on: bool) -> str:
        """Turn the UV fluorescence LED on or off."""
        return self.arduino.send_command(
            f"LED:UV:{'ON' if on else 'OFF'}", expect_prefix="OK:"
        )


class MeshroomInterface:
    """
    Backwards-compatible wrapper around the photogrammetry module.

    Args:
        skip_meshroom: When True, publish from existing exports or demo fallback.
    """

    def __init__(self, skip_meshroom: bool = False) -> None:
        self.skip_meshroom = skip_meshroom

    def process(self, object_id: str):
        """Run the mesh pipeline stage for one object ID."""
        from src.pipeline.mesh_stage import process_object_mesh

        result, _record = process_object_mesh(object_id, skip_meshroom=self.skip_meshroom)
        return result
