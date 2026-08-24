#!/usr/bin/env python3
"""Capture Raspberry Pi camera focus samples and rank them by sharpness."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
from picamera2 import Picamera2


def sharpness_score(image_path: Path) -> float:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Captured image is unreadable: {image_path}")
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def parse_resolution(value: str) -> tuple[int, int] | None:
    if value.lower() == "auto":
        return None
    width, height = value.lower().split("x", maxsplit=1)
    return int(width), int(height)


def main() -> int:
    parser = argparse.ArgumentParser(description="AXUM Pi camera focus and resolution test")
    parser.add_argument("--resolution", default="auto", help="auto or WIDTHxHEIGHT")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--interval", type=float, default=None, help="seconds between unattended captures")
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=Path("camera_quality_test"))
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be positive")

    camera = Picamera2()
    sensor_resolution = tuple(int(value) for value in camera.sensor_resolution)
    resolution = parse_resolution(args.resolution) or sensor_resolution
    config = camera.create_still_configuration(main={"size": resolution, "format": "RGB888"})
    camera.configure(config)
    camera.start()
    time.sleep(max(0.0, args.warmup))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Camera sensor maximum: {sensor_resolution[0]}x{sensor_resolution[1]}")
    print(f"Capturing at: {resolution[0]}x{resolution[1]}")

    results: list[tuple[Path, float]] = []
    try:
        for index in range(1, args.count + 1):
            if args.interval is None:
                input(f"[{index}/{args.count}] Adjust focus, then press Enter to capture...")
            elif index > 1:
                time.sleep(max(0.0, args.interval))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = args.output_dir / f"focus_{index:02d}_{timestamp}.jpg"
            camera.capture_file(str(path), name="main", format="jpeg")
            score = sharpness_score(path)
            results.append((path, score))
            print(f"{path.name}: sharpness={score:.1f}")
    finally:
        camera.stop()
        camera.close()

    print("\nSharpest first:")
    for path, score in sorted(results, key=lambda item: item[1], reverse=True):
        print(f"{score:10.1f}  {path}")
    print("Sharpness is a relative focus metric; inspect the full-size winner visually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
