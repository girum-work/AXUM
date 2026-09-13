"""
calibrate_camera.py — measure camera intrinsics from chessboard captures.

WHY THIS IS THE BLOCKER: ArUco gives you a marker's corners in the image for
free, but turning those into a distance and a heading needs the camera's focal
length and distortion. Nothing in this repository has ever measured them.
pi/tools/pi_aruco_benchmark.py uses fx = fy = 900 with zero distortion and says
so in a comment -- that is fine for timing detection, which is what it was for,
but any pose computed against invented intrinsics is invented too.

WHAT YOU NEED: a printed chessboard, flat on rigid backing. The default is a
9x6 board, meaning 9x6 INNER CORNERS, which is a 10x7 grid of squares. Measure
one square with calipers and pass it; the number sets the unit of every
distance the rover later reports, so a 2% error there is a 2% error in range.

CAPTURE ADVICE, in order of how much it matters:
    1. Fill the frame differently each time -- near, far, all four corners.
       Views that only differ by translation add nothing.
    2. TILT the board, hard, 30-45 degrees. Focal length and distortion are
       only separable from oblique views; a stack of head-on shots produces a
       confident, wrong answer.
    3. 20 or more accepted views. Fewer will converge and will not generalise.
    4. Keep it sharp. Motion blur moves corners sub-pixel and biases the fit.

Reprojection error is reported because it is the only honest quality signal
here: it is the residual, in pixels, between where the fit says corners should
land and where they were found. Below ~0.5px is good for a webcam. Above ~1.0px
means something is wrong -- usually blur, a non-flat board, or a wrong square
size -- and the result should not be trusted.

Usage:
    python scripts/calibrate_camera.py --capture --camera 0 --square-mm 25
    python scripts/calibrate_camera.py --images data/calibration/front --square-mm 25
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def board_points(columns: int, rows: int, square_m: float) -> np.ndarray:
    """Corner positions in board coordinates, Z = 0."""
    points = np.zeros((columns * rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    return points * square_m


def find_corners(gray: np.ndarray, columns: int, rows: int):
    """Locate inner corners, refined to sub-pixel."""
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
             + cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(gray, (columns, rows), flags)
    if not found:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CORNER_CRITERIA)


def capture(grab, columns: int, rows: int, out_dir: Path,
            wanted: int) -> int:
    """
    Save frames in which the board is fully visible.

    Headless by necessity: requirements.txt pins opencv-python-headless, so
    there is no imshow anywhere in this codebase. Frames are graded as they
    arrive and the accepted ones written to disk, with progress on stdout.

    Args:
        grab: Callable returning the next frame, or None when exhausted.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Looking for a {columns}x{rows} inner-corner board. "
          f"Move and TILT it between captures; {wanted} accepted views wanted.")
    accepted, seen, last_saved = 0, 0, -999
    while accepted < wanted:
        frame = grab()
        if frame is None:
            break
        seen += 1
        gray = (frame if frame.ndim == 2
                else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        corners = find_corners(gray, columns, rows)
        # Space captures out, or 30 near-identical frames get saved while
        # the board sits still and the fit learns nothing from any of them.
        if corners is not None and seen - last_saved >= 15:
            path = out_dir / f"calib_{accepted:03d}.png"
            cv2.imwrite(str(path), frame)
            accepted += 1
            last_saved = seen
            print(f"  [{accepted}/{wanted}] saved {path.name}")

    print(f"\n{accepted} views in {out_dir}")
    return 0 if accepted else 1


def make_grabber(camera: int | None, stream_url: str | None):
    """
    Frame source for either a local device or one of the rover's cameras.

    Intrinsics belong to one camera at one resolution. The rover looks through
    an ESP32-CAM or the Pi IR-CUT camera, both HTTP streams -- calibrating the
    laptop webcam instead would produce a confident, wrong range on hardware,
    which is the exact failure marker_nav refuses to make.
    """
    if stream_url:
        from src.arm.controller import CameraInterface

        interface = CameraInterface(stream_url=stream_url)
        return interface.capture_array, lambda: None

    stream = cv2.VideoCapture(camera if camera is not None else 0)
    if not stream.isOpened():
        raise RuntimeError(f"Cannot open camera {camera}")

    def grab():
        ok, frame = stream.read()
        return frame if ok else None

    return grab, stream.release


def calibrate(image_dir: Path, columns: int, rows: int, square_m: float,
              report: Path) -> int:
    paths = sorted(p for p in image_dir.rglob("*")
                   if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        print(f"No images under {image_dir}", file=sys.stderr)
        return 1

    object_points, image_points, size, rejected = [], [], None, []
    template = board_points(columns, rows, square_m)
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        corners = find_corners(gray, columns, rows)
        if corners is None:
            rejected.append(path.name)
            continue
        object_points.append(template)
        image_points.append(corners)

    print(f"{len(image_points)} of {len(paths)} views usable")
    if rejected:
        print(f"  board not found in: {', '.join(rejected[:6])}"
              f"{' ...' if len(rejected) > 6 else ''}")
    if len(image_points) < 8:
        print("Need at least 8 usable views; 20+ is better.", file=sys.stderr)
        return 1

    error, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, size, None, None)

    per_view = []
    for index in range(len(object_points)):
        projected, _ = cv2.projectPoints(object_points[index], rvecs[index],
                                         tvecs[index], matrix, distortion)
        per_view.append(float(cv2.norm(image_points[index], projected,
                                       cv2.NORM_L2) / len(projected)))

    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    hfov = 2 * np.degrees(np.arctan(size[0] / (2 * fx)))
    vfov = 2 * np.degrees(np.arctan(size[1] / (2 * fy)))

    print(f"\nresolution        : {size[0]}x{size[1]}")
    print(f"focal length px   : fx {fx:.1f}  fy {fy:.1f}")
    print(f"principal point   : cx {cx:.1f}  cy {cy:.1f}  "
          f"(image centre {size[0] / 2:.0f}, {size[1] / 2:.0f})")
    print(f"field of view     : {hfov:.1f} deg horizontal, {vfov:.1f} vertical")
    print(f"distortion k1,k2  : {distortion.ravel()[0]:+.4f}, "
          f"{distortion.ravel()[1]:+.4f}")
    print(f"reprojection error: {error:.3f} px mean, "
          f"{max(per_view):.3f} worst view")

    if error > 1.0:
        print("\nWARNING: above 1.0px. Do not trust ranges from this. Usual "
              "causes are motion blur, a board that is not flat, or a wrong "
              "--square-mm.")
    elif error > 0.5:
        print("\nUsable, but more oblique views would tighten it.")

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "captured": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image_dir": str(image_dir),
        "views_used": len(image_points),
        "views_offered": len(paths),
        "resolution": list(size),
        "square_m": square_m,
        "camera_matrix": matrix.tolist(),
        "distortion": distortion.ravel().tolist(),
        "reprojection_error_px": float(error),
        "worst_view_error_px": max(per_view),
        "hfov_deg": float(hfov),
        "vfov_deg": float(vfov),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {report}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true",
                        help="Grab views from a camera first")
    parser.add_argument("--camera", type=int, default=None,
                        help="Local device index. Use --stream-url for the "
                             "rover's own cameras.")
    parser.add_argument("--stream-url",
                        help="ESP32_CAM_URL or PI_CAM_URL from config.py. "
                             "Calibrate the camera that will actually dock.")
    parser.add_argument("--images", type=Path,
                        default=Path("data/calibration/front"))
    parser.add_argument("--columns", type=int, default=9,
                        help="INNER corners across, not squares")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--square-mm", type=float, required=True,
                        help="Measured square edge. Sets the unit of every "
                             "distance the rover reports afterwards.")
    parser.add_argument("--views", type=int, default=20)
    parser.add_argument("--out", type=Path,
                        default=Path("data/calibration/front_intrinsics.json"))
    args = parser.parse_args()

    if args.capture:
        if args.camera is None and not args.stream_url:
            print("Pass --stream-url for a rover camera, or --camera for a "
                  "local one. Intrinsics do not transfer between cameras.",
                  file=sys.stderr)
            return 1
        try:
            grab, release = make_grabber(args.camera, args.stream_url)
        except RuntimeError as error:
            print(error, file=sys.stderr)
            return 1
        try:
            code = capture(grab, args.columns, args.rows, args.images,
                           args.views)
        finally:
            release()
        if code:
            return code
    return calibrate(args.images, args.columns, args.rows,
                     args.square_mm / 1000.0, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
