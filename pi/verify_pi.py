#!/usr/bin/env python3
"""Verify AXUM Raspberry Pi camera hardware and HTTP service behavior."""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import requests
from PIL import Image
from PIL.ExifTags import IFD, Base


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str
    warning: bool = False


def run_check(name: str, function) -> CheckResult:
    try:
        ok, message, warning = function()
        return CheckResult(name, bool(ok), str(message), bool(warning))
    except Exception as exc:
        return CheckResult(name, False, str(exc))


def check_imports() -> tuple[bool, str, bool]:
    try:
        from picamera2 import Picamera2  # noqa: F401
    except ImportError as exc:
        return False, f"picamera2 unavailable: {exc}", False
    return True, f"OpenCV {cv2.__version__}; Picamera2 import OK", False


def check_sensor_enumeration() -> tuple[bool, str, bool]:
    from picamera2 import Picamera2

    cameras = Picamera2.global_camera_info()
    if not cameras:
        return False, "No libcamera sensors detected", False
    models = [str(camera.get("Model") or camera.get("Id") or "unknown") for camera in cameras]
    return True, f"Detected {len(cameras)} camera(s): {', '.join(models)}", False


def check_camera_utility() -> tuple[bool, str, bool]:
    utility = shutil.which("rpicam-hello") or shutil.which("libcamera-hello")
    return bool(utility), utility or "Neither rpicam-hello nor libcamera-hello is installed", False


def decode_jpeg(payload: bytes) -> np.ndarray:
    if not payload.startswith(b"\xff\xd8"):
        raise ValueError("response does not start with JPEG SOI bytes")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode JPEG response")
    return image


def verify_service(base_url: str, captures: int, timeout: float) -> list[CheckResult]:
    base_url = base_url.rstrip("/")
    session = requests.Session()
    results: list[CheckResult] = []

    status_payload: dict = {}

    def status_check() -> tuple[bool, str, bool]:
        nonlocal status_payload
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                response = session.get(f"{base_url}/status", timeout=min(2.0, timeout))
                status_payload = response.json()
                response.raise_for_status()
                if status_payload.get("ok") is True:
                    return True, str(status_payload), False
            except Exception as exc:
                last_error = exc
            time.sleep(0.25)
        return False, str(last_error or status_payload or "service did not become ready"), False

    results.append(run_check("HTTP status", status_check))
    if not results[-1].ok:
        return results

    expected = status_payload.get("capture_resolution")

    def still_check() -> tuple[bool, str, bool]:
        dimensions = []
        for _ in range(captures):
            response = session.get(f"{base_url}/capture", timeout=timeout)
            response.raise_for_status()
            if "image/jpeg" not in response.headers.get("content-type", ""):
                raise ValueError(f"unexpected content type: {response.headers.get('content-type')}")
            image = decode_jpeg(response.content)
            dimensions.append([int(image.shape[1]), int(image.shape[0])])
        if expected and any(item != expected for item in dimensions):
            return False, f"Expected {expected}; got {dimensions}", False
        return True, f"{captures} JPEG capture(s), dimensions={dimensions[0]}", False

    results.append(run_check("Full-resolution stills", still_check))

    stream_ready = threading.Event()
    stream_result: dict[str, object] = {}

    def read_stream() -> None:
        try:
            with requests.get(f"{base_url}/stream", stream=True, timeout=(timeout, timeout)) as response:
                response.raise_for_status()
                stream_ready.set()
                payload = b""
                for chunk in response.iter_content(chunk_size=8192):
                    payload += chunk
                    start = payload.find(b"\xff\xd8")
                    end = payload.find(b"\xff\xd9", start + 2)
                    if start >= 0 and end >= 0:
                        image = decode_jpeg(payload[start:end + 2])
                        stream_result["dimensions"] = [int(image.shape[1]), int(image.shape[0])]
                        return
                    if len(payload) > 5_000_000:
                        raise ValueError("MJPEG frame boundary not found")
        except Exception as exc:
            stream_result["error"] = str(exc)
            stream_ready.set()

    thread = threading.Thread(target=read_stream, daemon=True)
    thread.start()
    stream_ready.wait(timeout)

    def concurrent_check() -> tuple[bool, str, bool]:
        response = session.get(f"{base_url}/capture", timeout=timeout)
        response.raise_for_status()
        image = decode_jpeg(response.content)
        thread.join(timeout)
        if thread.is_alive():
            return False, "Preview stream did not yield a complete JPEG frame", False
        if "error" in stream_result:
            return False, str(stream_result["error"]), False
        dimensions = [int(image.shape[1]), int(image.shape[0])]
        return True, f"Still={dimensions}; preview={stream_result.get('dimensions')}", False

    results.append(run_check("Concurrent stream and still", concurrent_check))

    def optics_check() -> tuple[bool, str, bool]:
        response = session.get(f"{base_url}/capture", timeout=timeout)
        response.raise_for_status()
        optics = Image.open(io.BytesIO(response.content)).getexif().get_ifd(IFD.Exif)
        focal = optics.get(Base.FocalLength)
        if not status_payload.get("exif_optics"):
            return True, ("no lens focal length set, so reconstruction will guess "
                          "the field of view -- set AXUM_LENS_FOCAL_MM"), True
        if not focal:
            return False, "status claims EXIF optics but the JPEG carries none", False
        return True, (f"focal {float(focal):.2f}mm, "
                      f"35mm equivalent {optics.get(Base.FocalLengthIn35mmFilm)}"), False

    results.append(run_check("EXIF optics", optics_check))

    def lock_check() -> tuple[bool, str, bool]:
        """Exposure drift only shows up in the finished mesh, so prove it here."""
        response = session.post(f"{base_url}/lock", json={"locked": True}, timeout=timeout)
        payload = response.json()
        if response.status_code != 200:
            return False, str(payload.get("error") or payload), False
        try:
            reported = session.get(f"{base_url}/status", timeout=timeout).json()
        finally:
            session.post(f"{base_url}/lock", json={"locked": False}, timeout=timeout)
        if not reported.get("capture_locked"):
            return False, "lock accepted but /status does not report it", False
        return True, f"pinned {', '.join(sorted(payload.get('controls') or {}))}", False

    results.append(run_check("Exposure and white balance lock", lock_check))
    results.append(CheckResult(
        "IR-cut control",
        True,
        f"{status_payload.get('ir_control')}; verify day/night switching physically before multispectral use",
        warning=True,
    ))
    return results


def print_results(results: list[CheckResult]) -> int:
    failures = 0
    for result in results:
        if result.warning:
            label = "WARN"
        elif result.ok:
            label = "PASS"
        else:
            label = "FAIL"
            failures += 1
        print(f"{label:<4} {result.name:<30} {result.message}")
    print(f"\n{len(results) - failures}/{len(results)} required checks passed")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify AXUM Pi camera hardware and service")
    parser.add_argument("--url", default="http://127.0.0.1:5001")
    parser.add_argument("--captures", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--direct-only", action="store_true")
    parser.add_argument("--service-only", action="store_true")
    args = parser.parse_args()

    if args.direct_only and args.service_only:
        parser.error("--direct-only and --service-only are mutually exclusive")

    results: list[CheckResult] = []
    if not args.service_only:
        results.extend([
            run_check("Camera utilities", check_camera_utility),
            run_check("Python camera imports", check_imports),
            run_check("Sensor enumeration", check_sensor_enumeration),
        ])
    if not args.direct_only:
        results.extend(verify_service(args.url, max(1, args.captures), args.timeout))
    return print_results(results)


if __name__ == "__main__":
    raise SystemExit(main())
