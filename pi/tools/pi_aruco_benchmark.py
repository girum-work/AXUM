"""
AXUM ROVER - Pi ArUco processing headroom benchmark.

WHAT: Measures how long ArUco detection + pose estimation actually takes,
so "can the Pi sustain ~10Hz alongside relay duties" has a real number
behind it instead of a guess. Answers Electronics' Item 4.

WHY runnable without the real camera: the front webcam is currently with
a teammate and unreachable (per Electronics' reply). This script doesn't
need it — it generates a synthetic ArUco marker image and measures the
CPU cost of detection + solvePnP on that, which is what actually matters
for a headroom question. Point it at a real camera with --camera when
the hardware's back, but don't block the profiling question on that.

USAGE:
    python pi_aruco_benchmark.py                    # synthetic image, no load simulation
    python pi_aruco_benchmark.py --simulate-load     # + background CPU contention
    python pi_aruco_benchmark.py --camera 0          # use a real camera instead
    python pi_aruco_benchmark.py --iterations 200

CAVEAT: --simulate-load is a rough approximation (a busy-loop thread), not
a substitute for profiling with the actual Pi relay/coordination code
running. Good enough for a go/no-go sanity check, not final sign-off.
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time

import cv2
import numpy as np

TARGET_HZ = 10
TARGET_FRAME_BUDGET_MS = 1000 / TARGET_HZ


def make_synthetic_frame(marker_px: int = 220, image_size: int = 720) -> np.ndarray:
    """Generate a test frame with one ArUco marker, roughly centered, for CPU-cost testing."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, 1, marker_px)
    frame = np.full((image_size, image_size, 3), 200, dtype=np.uint8)
    offset = (image_size - marker_px) // 2
    frame[offset:offset + marker_px, offset:offset + marker_px] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    # mild noise so this isn't a trivially easy, unrealistically clean detection
    noise = np.random.default_rng(0).normal(0, 5, frame.shape).astype(np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def busy_loop(stop_flag: list[bool]) -> None:
    """Rough approximation of background relay-duty CPU contention."""
    while not stop_flag[0]:
        _ = sum(i * i for i in range(2000))


def run_benchmark(frame_source, iterations: int, marker_length_m: float = 0.08) -> list[float]:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    # Placeholder camera intrinsics — this benchmark measures CPU cost, not
    # pose accuracy, so these don't need to be the real calibration values.
    image_size = 720
    camera_matrix = np.array([[900, 0, image_size / 2], [0, 900, image_size / 2], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((1, 5), dtype=np.float32)
    half = marker_length_m / 2
    object_points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)

    timings_ms = []
    for _ in range(iterations):
        frame = frame_source()
        start = time.perf_counter()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None and len(corners) > 0:
            cv2.solvePnP(object_points, corners[0], camera_matrix, dist_coeffs)

        elapsed_ms = (time.perf_counter() - start) * 1000
        timings_ms.append(elapsed_ms)
    return timings_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Pi ArUco processing headroom benchmark")
    parser.add_argument("--camera", type=int, default=None, help="camera index to use instead of a synthetic frame")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--simulate-load", action="store_true", help="run a background CPU-contention thread during the benchmark")
    args = parser.parse_args()

    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera index {args.camera}")

        def frame_source():
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera read failed mid-benchmark")
            return frame
    else:
        static_frame = make_synthetic_frame()
        frame_source = lambda: static_frame

    stop_flag = [False]
    load_threads = []
    if args.simulate_load:
        print("Starting simulated background load (4 threads)...")
        for _ in range(4):
            t = threading.Thread(target=busy_loop, args=(stop_flag,), daemon=True)
            t.start()
            load_threads.append(t)

    print(f"Running {args.iterations} iterations "
          f"({'real camera ' + str(args.camera) if args.camera is not None else 'synthetic frame'}, "
          f"{'with' if args.simulate_load else 'without'} simulated load)...")

    timings_ms = run_benchmark(frame_source, args.iterations)

    stop_flag[0] = True
    for t in load_threads:
        t.join(timeout=1)

    mean_ms = statistics.mean(timings_ms)
    p95_ms = statistics.quantiles(timings_ms, n=20)[18] if len(timings_ms) >= 20 else max(timings_ms)
    max_ms = max(timings_ms)
    achievable_hz = 1000 / mean_ms if mean_ms > 0 else float("inf")

    print()
    print(f"Mean latency:     {mean_ms:.2f} ms")
    print(f"P95 latency:      {p95_ms:.2f} ms")
    print(f"Max latency:      {max_ms:.2f} ms")
    print(f"Target budget:    {TARGET_FRAME_BUDGET_MS:.2f} ms/frame (for {TARGET_HZ}Hz)")
    print(f"Mean-case achievable rate: ~{achievable_hz:.1f} Hz")
    print()
    if p95_ms < TARGET_FRAME_BUDGET_MS:
        print(f"RESULT: P95 latency fits within the {TARGET_HZ}Hz budget — looks achievable.")
    elif mean_ms < TARGET_FRAME_BUDGET_MS:
        print(f"RESULT: Mean fits but P95 doesn't — {TARGET_HZ}Hz is marginal, expect occasional missed cycles.")
    else:
        print(f"RESULT: Mean latency alone exceeds the {TARGET_HZ}Hz budget — {TARGET_HZ}Hz is not realistic as specified, "
              f"consider a lower target rate or reducing detection resolution.")


if __name__ == "__main__":
    main()