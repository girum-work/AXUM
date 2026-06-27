# src/imaging/salt_mapper.py
"""
AXUM ROVER — Salt Crystallization Mapper
==========================================
Detects and maps salt deposits on stone artefact surfaces using
UV fluorescence imaging and electrical conductivity sensing.

HOW IT WORKS (plain English):
    Salt weathering is the #1 cause of inscription destruction that
    nobody talks about. Here's what happens:

        Stone absorbs water → water carries dissolved salts →
        water evaporates → salts crystallize inside stone pores →
        crystals EXPAND as they grow → expansion shatters stone
        from inside out → inscription destroyed

    The problem: salt deposits are often invisible to normal light.
    The solution: under 365nm UV light, many salt compounds
    fluoresce — they GLOW white or yellow-green. By capturing a
    UV image and detecting these glowing regions, we can map exactly
    where salt is accumulating BEFORE it causes visible damage.

    We combine two detection methods:
    1. UV fluorescence (camera): finds surface salt deposits
    2. Conductivity probe (sensor): confirms salt presence in the
       stone itself (not just surface dust)

    Together they give a salt risk map — where is salt accumulating,
    how much is there, and is it actively migrating through the stone?

HARDWARE USED:
    - 365nm UV LEDs ×6 (same as multispectral module)
      Arduino pin 37 → MOSFET → UV LED bank
    - ESP32-CAM (captures UV fluorescence image)
    - Conductivity probe (2 stainless steel pins, 5mm apart)
      Connected to STM32F411 12-bit ADC via I2C
      Arduino triggers measurement via: CONDUCT command
    - The probe is mounted on the arm tip, pressed lightly
      against the stone surface at grid points

SALT RISK LEVELS:
    Level 0 — No salt detected
    Level 1 — Trace fluorescence, no conductivity signal
              (surface dust, not a threat)
    Level 2 — Moderate fluorescence + low conductivity
              (early salt accumulation, monitor)
    Level 3 — Strong fluorescence + high conductivity
              (active salt migration, intervention needed)
    Level 4 — Saturation fluorescence + very high conductivity
              (critical — inscription at immediate risk)

Author: Axum Rover Team
"""

import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field
from typing import Optional
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import SCAN_PHOTOS_DIR


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class SaltZone:
    """
    One detected salt accumulation zone.

    Attributes:
        bbox:          (x, y, w, h) bounding box in image coordinates
        center:        (cx, cy) centre pixel
        area_px:       Area in pixels
        fluorescence:  float 0–1, UV fluorescence intensity
        conductivity:  float 0–1, normalised conductivity reading
                       (0 if probe not available)
        risk_level:    int 0–4 (see module docstring)
        is_near_inscription: bool — True if within 50px of OCR text region
        migration_vector: (dx, dy) or None — estimated direction of
                          salt migration based on gradient
    """
    bbox:               tuple
    center:             tuple
    area_px:            int
    fluorescence:       float
    conductivity:       float = 0.0
    risk_level:         int   = 0
    is_near_inscription: bool = False
    migration_vector:   tuple = None


@dataclass
class SaltMapResult:
    """
    Complete output from one salt mapping scan.

    Attributes:
        visible_image:     (H, W, 3) uint8 BGR reference image
        uv_image:          (H, W) float32 UV fluorescence 0–1
        fluorescence_map:  (H, W) float32 — normalised salt fluorescence
        conductivity_grid: list of (x, y, value) measurements from probe
        salt_mask:         (H, W) uint8 binary — 255 = salt detected
        risk_map:          (H, W) uint8 — 0–4 risk level per pixel
        salt_zones:        list of SaltZone, sorted by risk (worst first)
        overall_risk:      float 0–1, weighted surface risk score
        critical_zones:    list of SaltZone with risk_level >= 3
        migration_paths:   list of line segments showing salt movement
        quality_score:     float 0–1
    """
    visible_image:      np.ndarray
    uv_image:           np.ndarray
    fluorescence_map:   np.ndarray
    conductivity_grid:  list
    salt_mask:          np.ndarray
    risk_map:           np.ndarray
    salt_zones:         list
    overall_risk:       float
    critical_zones:     list
    migration_paths:    list
    quality_score:      float


# Thresholds — tune after testing on real artefacts
FLUOR_THRESHOLD       = 0.45   # UV brightness above which = fluorescence
CONDUCTIVITY_THRESHOLD = 0.30  # Normalised conductivity above which = salt
MIN_SALT_AREA_PX      = 30     # Minimum zone area to report
INSCRIPTION_MARGIN_PX = 50     # Distance within which salt is near inscription


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — HARDWARE CAPTURE
# ═══════════════════════════════════════════════════════════════

def capture_uv_fluorescence(
    arduino,
    camera_url: str,
    settle_ms:  int = 300
) -> tuple:
    """
    Capture UV fluorescence image of the artefact surface.

    Procedure:
        1. Turn off all lights (dark background essential for fluorescence)
        2. Turn on UV LEDs only
        3. Wait for LEDs to stabilise (longer than normal — UV LEDs
           need ~300ms to reach full output)
        4. Capture image
        5. Turn all lights off

    Why darkness matters:
        Salt fluorescence is faint. Any ambient or white light will
        overwhelm the fluorescence signal. The room should be as dark
        as possible during capture. The turntable enclosure helps.

    Args:
        arduino:    ArduinoSerial instance
        camera_url: http://<ESP32_IP>/capture
        settle_ms:  LED warm-up time (300ms recommended for UV)

    Returns:
        (visible_bgr, uv_gray) tuple
        visible_bgr: reference image under white light
        uv_gray: (H, W) float32 UV fluorescence image
    """
    import requests

    def capture(label):
        try:
            r         = requests.get(camera_url, timeout=3)
            arr       = np.frombuffer(r.content, dtype=np.uint8)
            img       = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                logger.error(f"Failed to decode {label} image")
                return None
            return img
        except Exception as e:
            logger.error(f"Capture failed [{label}]: {e}")
            return None

    # ── Visible reference ──────────────────────────────────────
    arduino.send_command("LED:READY")
    time.sleep(0.15)
    visible = capture("visible")
    arduino.send_command("LED:OFF")
    time.sleep(0.1)

    # ── UV fluorescence ────────────────────────────────────────
    # Ensure complete darkness before UV capture
    arduino.send_command("LED:OFF")
    time.sleep(0.2)
    arduino.send_command("LED:UV:ON")
    time.sleep(settle_ms / 1000.0)

    uv_bgr = capture("UV fluorescence")
    arduino.send_command("LED:OFF")

    if uv_bgr is None:
        return visible, None

    # Convert UV image to grayscale float
    uv_gray = cv2.cvtColor(uv_bgr, cv2.COLOR_BGR2GRAY)\
                .astype(np.float32) / 255.0

    return visible, uv_gray


def measure_conductivity_grid(
    arduino,
    grid_points: list,
    press_depth_mm: float = 1.0
) -> list:
    """
    Measure surface conductivity at a grid of points using the probe.

    The conductivity probe is two stainless steel pins mounted on
    the arm tip, 5mm apart. When pressed against a moist or
    salt-contaminated stone surface, current flows between them.
    More salt = more ions = more conductivity.

    The probe is moved to each grid point by the arm controller,
    pressed lightly against the surface, and a reading is taken.

    Args:
        arduino:        ArduinoSerial instance
        grid_points:    List of (arm_x, arm_y) positions for probe
                        These are arm workspace coordinates, not pixels.
                        Generate a 3×3 or 4×4 grid across the artefact.
        press_depth_mm: How far to press probe into surface.
                        1mm is safe for most stone artefacts.

    Returns:
        measurements: List of (x, y, conductivity_value) tuples
                      conductivity_value is raw ADC reading 0–4095
                      (12-bit from STM32 co-processor)
    """
    measurements = []

    for i, (ax, ay) in enumerate(grid_points):
        logger.debug(f"Conductivity probe: point {i+1}/{len(grid_points)} "
                    f"at ({ax:.1f}, {ay:.1f})")

        # Move arm to probe position
        # Note: arm coordinates depend on your specific arm calibration
        # This assumes the probe is at the wrist position
        try:
            arduino.send_command(f"ARM:PROBE:{ax:.1f},{ay:.1f}")
            time.sleep(0.5)  # wait for arm to settle

            # Request conductivity reading from STM32 via Arduino
            response = arduino.send_command("CONDUCT")

            # Parse response: "CONDUCT:2048" → 2048
            if response and response.startswith("CONDUCT:"):
                value = int(response.split(":")[1])
                measurements.append((ax, ay, value))
                logger.debug(f"  Reading: {value} (raw ADC)")
            else:
                logger.warning(f"  No reading at ({ax}, {ay})")
                measurements.append((ax, ay, 0))

        except Exception as e:
            logger.error(f"Probe measurement failed at ({ax}, {ay}): {e}")
            measurements.append((ax, ay, 0))

    # Return arm to home position
    arduino.send_command("POSE:HOME")

    return measurements


def conductivity_grid_to_image(
    measurements: list,
    image_shape:  tuple,
    artefact_bbox: tuple = None
) -> np.ndarray:
    """
    Interpolate sparse conductivity grid measurements into a full image.

    The probe only measures at N grid points, but we want a continuous
    conductivity map to overlay on the camera image. We use bilinear
    interpolation to fill in between measurement points.

    Args:
        measurements:   List of (x, y, value) from measure_conductivity_grid
        image_shape:    (H, W) of the camera image
        artefact_bbox:  (x, y, w, h) of artefact in image, or None

    Returns:
        conductivity_map: (H, W) float32, normalised 0–1
    """
    if not measurements:
        return np.zeros(image_shape[:2], dtype=np.float32)

    h, w = image_shape[:2]

    # Extract coordinates and values
    xs     = np.array([m[0] for m in measurements], dtype=np.float32)
    ys     = np.array([m[1] for m in measurements], dtype=np.float32)
    values = np.array([m[2] for m in measurements], dtype=np.float32)

    # Normalise values to 0–1
    v_max = values.max()
    if v_max > 0:
        values = values / v_max

    # Map arm coordinates to image coordinates
    # Assumes arm workspace maps linearly to image space
    if artefact_bbox is not None:
        bx, by, bw, bh = artefact_bbox
        xs_px = bx + (xs - xs.min()) / (xs.max() - xs.min() + 1e-6) * bw
        ys_px = by + (ys - ys.min()) / (ys.max() - ys.min() + 1e-6) * bh
    else:
        xs_px = (xs - xs.min()) / (xs.max() - xs.min() + 1e-6) * w
        ys_px = (ys - ys.min()) / (ys.max() - ys.min() + 1e-6) * h

    # Create sparse map and interpolate
    sparse_map = np.zeros((h, w), dtype=np.float32)
    for px, py, val in zip(xs_px.astype(int), ys_px.astype(int), values):
        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)
        sparse_map[py, px] = val

    # Interpolate using distance-weighted blur
    # Where we have no measurements, blur from nearby points
    kernel_size = max(h, w) // 4
    if kernel_size % 2 == 0:
        kernel_size += 1
    conductivity_map = cv2.GaussianBlur(sparse_map, (kernel_size, kernel_size), 0)

    # Renormalise after blur
    c_max = conductivity_map.max()
    if c_max > 0:
        conductivity_map /= c_max

    return conductivity_map


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — CORE SALT DETECTION
# ═══════════════════════════════════════════════════════════════

def extract_fluorescence_map(uv_image: np.ndarray) -> np.ndarray:
    """
    Extract salt fluorescence signal from UV image.

    Under UV light, the camera captures everything — stone background,
    organic material, AND salt fluorescence. We need to isolate
    the salt signal.

    Salt fluorescence characteristics:
        - Brighter than stone background under UV
        - Often bluish-white or yellow-green tint
        - Appears in clusters along crack lines or pore networks
        - More uniform than specular reflections (which are sharp spots)

    Processing steps:
        1. Remove background stone UV response (subtract rolling average)
        2. Threshold bright regions (= fluorescence candidates)
        3. Remove specular reflections (sharp, isolated bright spots)
        4. What remains = likely salt fluorescence

    Args:
        uv_image: (H, W) float32 UV image 0–1

    Returns:
        fluorescence_map: (H, W) float32 salt fluorescence strength 0–1
    """
    # Step 1: Estimate stone background UV response
    # Use a large blur as the "expected" background level
    background = cv2.GaussianBlur(uv_image, (61, 61), 0)

    # Step 2: Subtract background — what's left is local bright spots
    foreground = np.clip(uv_image - background, 0, None)

    # Step 3: Remove specular reflections
    # Specular = very small, very bright isolated pixels
    # Salt fluorescence = larger, more diffuse patches
    # Use morphological opening to remove tiny bright spots
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(
        (foreground * 255).astype(np.uint8),
        cv2.MORPH_OPEN, kernel
    ).astype(np.float32) / 255.0

    # Step 4: Smooth to get continuous fluorescence regions
    fluorescence_map = cv2.GaussianBlur(cleaned, (7, 7), 0)

    # Normalise to 0–1
    f_max = fluorescence_map.max()
    if f_max > 0:
        fluorescence_map /= f_max

    return fluorescence_map


def compute_risk_map(
    fluorescence_map:  np.ndarray,
    conductivity_map:  np.ndarray = None,
    fluor_thresh:      float = FLUOR_THRESHOLD,
    conduct_thresh:    float = CONDUCTIVITY_THRESHOLD
) -> np.ndarray:
    """
    Compute per-pixel risk level (0–4) from fluorescence and conductivity.

    Risk level assignment:
        0: fluorescence < thresh AND conductivity < thresh
           → No salt detected
        1: fluorescence >= thresh AND conductivity < thresh
           → Surface fluorescence only (could be dust, organic material)
        2: fluorescence >= thresh AND conductivity >= thresh (low)
           → Confirmed salt, early accumulation
        3: fluorescence >= thresh AND conductivity >= thresh (moderate)
           → Active salt migration
        4: fluorescence saturated AND conductivity high
           → Critical — immediate intervention needed

    Args:
        fluorescence_map:  (H, W) float32 0–1
        conductivity_map:  (H, W) float32 0–1, or None if no probe data
        fluor_thresh:      Fluorescence threshold for detection
        conduct_thresh:    Conductivity threshold for confirmation

    Returns:
        risk_map: (H, W) uint8 values 0–4
    """
    h, w     = fluorescence_map.shape
    risk_map = np.zeros((h, w), dtype=np.uint8)

    # Fluorescence mask
    fluor_mask = fluorescence_map >= fluor_thresh

    if conductivity_map is None:
        # No probe data — use fluorescence only
        # Risk 1 for low fluorescence, risk 2 for high fluorescence
        risk_map[fluor_mask & (fluorescence_map < 0.7)] = 1
        risk_map[fluor_mask & (fluorescence_map >= 0.7)] = 2
    else:
        # Ensure same size
        if conductivity_map.shape != (h, w):
            conductivity_map = cv2.resize(conductivity_map, (w, h))

        conduct_low  = (conductivity_map >= conduct_thresh) & \
                       (conductivity_map < 0.6)
        conduct_high = conductivity_map >= 0.6

        # Level 1: fluorescence only
        risk_map[fluor_mask & ~conduct_low & ~conduct_high] = 1

        # Level 2: fluorescence + low conductivity
        risk_map[fluor_mask & conduct_low] = 2

        # Level 3: fluorescence + high conductivity
        risk_map[fluor_mask & conduct_high] = 3

        # Level 4: saturated fluorescence + very high conductivity
        saturation = (fluorescence_map >= 0.85) & (conductivity_map >= 0.75)
        risk_map[saturation] = 4

    return risk_map


def find_salt_zones(
    risk_map:          np.ndarray,
    fluorescence_map:  np.ndarray,
    conductivity_map:  np.ndarray = None,
    inscription_mask:  np.ndarray = None
) -> list:
    """
    Find individual salt accumulation zones from the risk map.

    Each connected region of risk > 0 becomes one SaltZone.

    Args:
        risk_map:         (H, W) uint8 0–4 risk levels
        fluorescence_map: (H, W) float32 for severity measurement
        conductivity_map: (H, W) float32 or None
        inscription_mask: (H, W) uint8 binary — where text regions are
                          Used to flag zones near inscriptions.
                          Pass the crack_mask from photometric stereo
                          or a text region mask from OCR pipeline.

    Returns:
        zones: List of SaltZone, sorted by risk_level descending
    """
    # Find connected components of any risk > 0
    binary_risk = (risk_map > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary_risk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    zones = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_SALT_AREA_PX:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        cx = int(x + bw / 2)
        cy = int(y + bh / 2)

        # Create zone mask
        zone_mask = np.zeros(risk_map.shape, dtype=np.uint8)
        cv2.drawContours(zone_mask, [cnt], -1, 255, -1)

        # Measure fluorescence in this zone
        fluor_vals   = fluorescence_map[zone_mask > 0]
        fluorescence = float(fluor_vals.mean()) if len(fluor_vals) > 0 else 0.0

        # Measure conductivity if available
        conductivity = 0.0
        if conductivity_map is not None:
            cond_vals    = conductivity_map[zone_mask > 0]
            conductivity = float(cond_vals.mean()) if len(cond_vals) > 0 else 0.0

        # Risk level = max risk in this zone
        risk_vals  = risk_map[zone_mask > 0]
        risk_level = int(risk_vals.max()) if len(risk_vals) > 0 else 0

        # Check proximity to inscription
        is_near = False
        if inscription_mask is not None:
            # Dilate inscription mask by INSCRIPTION_MARGIN_PX
            margin_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (INSCRIPTION_MARGIN_PX * 2, INSCRIPTION_MARGIN_PX * 2)
            )
            insc_dilated = cv2.dilate(inscription_mask, margin_kernel)
            # Check if any salt zone pixel overlaps with dilated inscription
            overlap = cv2.bitwise_and(zone_mask, insc_dilated)
            is_near = bool(np.any(overlap > 0))

        # Estimate migration vector from fluorescence gradient
        migration_vec = None
        if area > 200:
            region_fluor = fluorescence_map[y:y+bh, x:x+bw]
            gy, gx = np.gradient(region_fluor)
            mean_gx = float(gx.mean())
            mean_gy = float(gy.mean())
            mag = np.sqrt(mean_gx**2 + mean_gy**2)
            if mag > 0.01:
                migration_vec = (mean_gx / mag, mean_gy / mag)

        zones.append(SaltZone(
            bbox               = (x, y, bw, bh),
            center             = (cx, cy),
            area_px            = int(area),
            fluorescence       = fluorescence,
            conductivity       = conductivity,
            risk_level         = risk_level,
            is_near_inscription = is_near,
            migration_vector   = migration_vec
        ))

    # Sort by risk level descending, then by area
    zones.sort(key=lambda z: (z.risk_level, z.area_px), reverse=True)
    return zones


def estimate_migration_paths(
    fluorescence_map: np.ndarray,
    salt_zones:       list
) -> list:
    """
    Estimate paths along which salt is migrating through the stone.

    Salt migrates along crack lines and pore networks. By following
    the gradient of the fluorescence map from high-concentration zones
    toward lower-concentration zones, we can estimate where the salt
    came from and where it's heading.

    This is a simplified gradient descent from each major zone.
    The output is a list of line segments for visualisation.

    Args:
        fluorescence_map: (H, W) float32
        salt_zones:       From find_salt_zones()

    Returns:
        paths: List of lists of (x, y) points — each inner list
               is one migration path
    """
    paths = []

    # Only trace paths for significant zones (risk >= 2)
    major_zones = [z for z in salt_zones if z.risk_level >= 2][:5]

    for zone in major_zones:
        cx, cy = zone.center
        path   = [(cx, cy)]

        # Follow fluorescence gradient for up to 50 steps
        x, y = float(cx), float(cy)
        h, w = fluorescence_map.shape

        for _ in range(50):
            xi, yi = int(x), int(y)
            if xi < 1 or xi >= w-1 or yi < 1 or yi >= h-1:
                break

            # Local gradient
            gx = float(fluorescence_map[yi, xi+1] - fluorescence_map[yi, xi-1])
            gy = float(fluorescence_map[yi+1, xi] - fluorescence_map[yi-1, xi])
            mag = np.sqrt(gx**2 + gy**2)

            if mag < 0.001:
                break

            # Step in gradient direction (toward higher concentration = source)
            step = 3.0
            x += (gx / mag) * step
            y += (gy / mag) * step

            new_pt = (int(x), int(y))
            if new_pt != path[-1]:
                path.append(new_pt)

        if len(path) > 3:
            paths.append(path)

    return paths


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — VISUALISATION
# ═══════════════════════════════════════════════════════════════

# Risk level colours (BGR)
RISK_COLOURS = {
    0: (80,  80,  80),   # grey   — no risk
    1: (0,   200, 255),  # yellow — trace
    2: (0,   140, 255),  # orange — early
    3: (0,   60,  255),  # red    — active
    4: (0,   0,   200),  # dark red — critical
}

RISK_LABELS = {
    0: "None",
    1: "Trace",
    2: "Early",
    3: "Active",
    4: "Critical"
}


def render_risk_map_colour(risk_map: np.ndarray) -> np.ndarray:
    """
    Render risk level map as a colour image.

    Each risk level gets a distinct colour:
        Grey → Yellow → Orange → Red → Dark Red

    Args:
        risk_map: (H, W) uint8 values 0–4

    Returns:
        colour: (H, W, 3) uint8 BGR image
    """
    h, w    = risk_map.shape
    colour  = np.zeros((h, w, 3), dtype=np.uint8)

    for level, bgr in RISK_COLOURS.items():
        mask = risk_map == level
        colour[mask] = bgr

    return colour


def render_salt_overlay(
    visible_image: np.ndarray,
    risk_map:      np.ndarray,
    salt_zones:    list,
    migration_paths: list = None
) -> np.ndarray:
    """
    Overlay salt risk information on the visible light image.

    Shows the artefact as it appears normally, with:
        - Semi-transparent colour overlay for risk zones
        - Bounding boxes around each salt zone
        - Risk level labels
        - Migration path arrows

    Args:
        visible_image:   (H, W, 3) uint8 BGR reference
        risk_map:        (H, W) uint8 0–4
        salt_zones:      From find_salt_zones()
        migration_paths: From estimate_migration_paths(), or None

    Returns:
        overlay: (H, W, 3) uint8 annotated image
    """
    overlay = visible_image.copy()
    h, w    = overlay.shape[:2]

    # Resize risk map if needed
    if risk_map.shape[:2] != (h, w):
        risk_map = cv2.resize(risk_map, (w, h),
                              interpolation=cv2.INTER_NEAREST)

    # Semi-transparent risk colour overlay (skip level 0)
    risk_colour_img = render_risk_map_colour(risk_map)
    non_zero_mask   = risk_map > 0
    overlay[non_zero_mask] = cv2.addWeighted(
        overlay, 0.5,
        risk_colour_img, 0.5,
        0
    )[non_zero_mask]

    # Draw zone bounding boxes and labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs   = 0.45

    for i, zone in enumerate(salt_zones[:8]):
        x, y, bw, bh = zone.bbox
        col          = RISK_COLOURS[zone.risk_level]

        cv2.rectangle(overlay, (x, y), (x+bw, y+bh), col, 2)

        label = f"S{i+1} L{zone.risk_level}:{RISK_LABELS[zone.risk_level]}"
        if zone.is_near_inscription:
            label += " !"  # alert marker

        # Background for label
        (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
        cv2.rectangle(overlay, (x, y-th-6), (x+tw+4, y), (0, 0, 0), -1)
        cv2.putText(overlay, label, (x+2, y-3), font, fs, col, 1)

        # Migration vector arrow
        if zone.migration_vector is not None:
            cx, cy   = zone.center
            dx, dy   = zone.migration_vector
            end_x    = int(cx + dx * 20)
            end_y    = int(cy + dy * 20)
            cv2.arrowedLine(overlay, (cx, cy), (end_x, end_y),
                           col, 2, tipLength=0.3)

    # Draw migration paths
    if migration_paths:
        for path in migration_paths:
            for j in range(1, len(path)):
                cv2.line(overlay, path[j-1], path[j], (255, 200, 0), 1)

    return overlay


def render_salt_composite(result: SaltMapResult) -> np.ndarray:
    """
    Create a 2×2 composite display of all salt mapping outputs.

    Layout:
        Top-left:     Visible light reference
        Top-right:    UV fluorescence (hot colourmap)
        Bottom-left:  Risk level map (colour coded)
        Bottom-right: Salt overlay on visible image

    Args:
        result: SaltMapResult

    Returns:
        composite: (H*2, W*2, 3) uint8
    """
    vis = result.visible_image
    if vis is None:
        h, w = result.risk_map.shape
        vis  = np.zeros((h, w, 3), dtype=np.uint8)

    h, w = vis.shape[:2]

    def resize(img):
        return cv2.resize(img, (w, h))

    panel_vis   = resize(vis)
    panel_uv    = resize(cv2.applyColorMap(
        (result.uv_image * 255).astype(np.uint8)
        if result.uv_image is not None
        else np.zeros((h, w), dtype=np.uint8),
        cv2.COLORMAP_HOT
    ))
    panel_risk  = resize(render_risk_map_colour(result.risk_map))
    panel_over  = resize(render_salt_overlay(
        vis, result.risk_map, result.salt_zones, result.migration_paths
    ))

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, col, th = 0.5, (255, 255, 255), 1

    cv2.putText(panel_vis,  "Visible Light",     (8, 20), font, fs, col, th)
    cv2.putText(panel_uv,   "UV Fluorescence",   (8, 20), font, fs, col, th)
    cv2.putText(panel_risk, "Salt Risk Map",      (8, 20), font, fs, col, th)
    cv2.putText(panel_over, "Salt Overlay",       (8, 20), font, fs, col, th)

    # Overall risk score
    risk_text = f"Overall risk: {result.overall_risk:.1%}"
    risk_col  = RISK_COLOURS[min(4, int(result.overall_risk * 4))]
    cv2.putText(panel_risk, risk_text, (8, h-8), font, fs, risk_col, th)

    # Critical zone count
    if result.critical_zones:
        crit_text = f"CRITICAL: {len(result.critical_zones)} zones"
        cv2.putText(panel_over, crit_text, (8, h-8), font, fs, (0, 0, 255), th)

    top    = np.hstack([panel_vis,  panel_uv])
    bottom = np.hstack([panel_risk, panel_over])
    return np.vstack([top, bottom])


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — MAIN PIPELINE ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def run_salt_mapper(
    visible_image:     np.ndarray = None,
    uv_image:          np.ndarray = None,
    conductivity_grid: list       = None,
    arduino=None,
    camera_url:        str        = None,
    image_dir:         Path       = None,
    inscription_mask:  np.ndarray = None,
    save_dir:          Path       = None,
    artefact_id:       str        = "unknown"
) -> SaltMapResult:
    """
    Run the full salt mapping pipeline.

    Three ways to provide images:
        1. Pass pre-loaded arrays directly
        2. Pass arduino + camera_url for live capture
        3. Pass image_dir to load from disk (testing)

    Args:
        visible_image:     (H, W, 3) uint8 BGR reference image
        uv_image:          (H, W) float32 UV image 0–1
        conductivity_grid: List of (x, y, value) from probe measurements
        arduino:           ArduinoSerial instance
        camera_url:        ESP32-CAM capture URL
        image_dir:         Directory with visible.png + uv.png
        inscription_mask:  (H, W) uint8 — text region mask from OCR
                           Pass this to flag salt zones near inscriptions
        save_dir:          Save outputs here if provided
        artefact_id:       Used for output filenames

    Returns:
        SaltMapResult

    Example (from main_pipeline.py):
        from src.imaging.salt_mapper import run_salt_mapper

        salt_result = run_salt_mapper(
            arduino          = self.arduino,
            camera_url       = self.camera_url,
            inscription_mask = ocr_result.text_mask,
            artefact_id      = artefact.id
        )

        critical = salt_result.critical_zones
        if critical:
            logger.warning(f"{len(critical)} critical salt zones detected")
    """
    logger.info(f"Starting salt mapping for: {artefact_id}")

    # ── Step 1: Get images ─────────────────────────────────────
    if visible_image is not None and uv_image is not None:
        logger.info("Using pre-loaded images")

    elif arduino is not None and camera_url is not None:
        logger.info("Capturing UV fluorescence from hardware...")
        visible_image, uv_image = capture_uv_fluorescence(
            arduino, camera_url
        )

    elif image_dir is not None:
        logger.info(f"Loading from disk: {image_dir}")
        image_dir   = Path(image_dir)
        vis_path    = image_dir / "visible.png"
        uv_path     = image_dir / "uv.png"
        visible_image = cv2.imread(str(vis_path)) if vis_path.exists() else None
        if uv_path.exists():
            uv_gray   = cv2.imread(str(uv_path), cv2.IMREAD_GRAYSCALE)
            uv_image  = uv_gray.astype(np.float32) / 255.0
        else:
            uv_image  = None

    else:
        raise ValueError("Must provide images, arduino+camera_url, or image_dir")

    quality = 1.0 if uv_image is not None else 0.0

    # Fallback visible
    if visible_image is None and uv_image is not None:
        visible_image = cv2.cvtColor(
            (uv_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
        )

    # Ensure consistent size
    if visible_image is not None and uv_image is not None:
        h, w = visible_image.shape[:2]
        if uv_image.shape != (h, w):
            uv_image = cv2.resize(uv_image, (w, h))

    h, w = (visible_image.shape[:2] if visible_image is not None
            else uv_image.shape if uv_image is not None
            else (480, 640))

    # ── Step 2: Extract fluorescence map ──────────────────────
    logger.info("Extracting fluorescence signal...")
    if uv_image is not None:
        fluorescence_map = extract_fluorescence_map(uv_image)
    else:
        fluorescence_map = np.zeros((h, w), dtype=np.float32)
        quality = 0.0

    # ── Step 3: Build conductivity map ────────────────────────
    if conductivity_grid:
        conductivity_map = conductivity_grid_to_image(
            conductivity_grid, (h, w)
        )
        logger.info(f"Conductivity grid: {len(conductivity_grid)} points")
    else:
        conductivity_map = None
        logger.info("No conductivity data — using fluorescence only")

    # ── Step 4: Compute risk map ───────────────────────────────
    logger.info("Computing risk map...")
    risk_map = compute_risk_map(fluorescence_map, conductivity_map)

    # ── Step 5: Find salt zones ────────────────────────────────
    salt_zones = find_salt_zones(
        risk_map, fluorescence_map,
        conductivity_map, inscription_mask
    )

    # ── Step 6: Estimate migration paths ──────────────────────
    migration_paths = estimate_migration_paths(fluorescence_map, salt_zones)

    # ── Step 7: Compute summary statistics ────────────────────
    # Overall risk = weighted average of risk levels
    if risk_map.size > 0:
        risk_weights  = np.array([0, 0.1, 0.3, 0.7, 1.0])
        pixel_risks   = np.array([
            np.sum(risk_map == level) / risk_map.size
            for level in range(5)
        ])
        overall_risk  = float(np.dot(pixel_risks, risk_weights))
    else:
        overall_risk  = 0.0

    critical_zones = [z for z in salt_zones if z.risk_level >= 3]
    near_insc      = [z for z in salt_zones if z.is_near_inscription]

    logger.info(f"Salt mapping complete: "
               f"{len(salt_zones)} zones, "
               f"{len(critical_zones)} critical, "
               f"{len(near_insc)} near inscriptions, "
               f"overall risk={overall_risk:.1%}")

    if critical_zones:
        logger.warning(f"CRITICAL SALT ZONES DETECTED: {len(critical_zones)}")
        for i, zone in enumerate(critical_zones[:3]):
            logger.warning(f"  Zone {i+1}: risk={zone.risk_level}, "
                          f"fluor={zone.fluorescence:.2f}, "
                          f"near_inscription={zone.is_near_inscription}")

    # ── Step 8: Package result ────────────────────────────────
    result = SaltMapResult(
        visible_image     = visible_image,
        uv_image          = uv_image,
        fluorescence_map  = fluorescence_map,
        conductivity_grid = conductivity_grid or [],
        salt_mask         = (risk_map > 0).astype(np.uint8) * 255,
        risk_map          = risk_map,
        salt_zones        = salt_zones,
        overall_risk      = overall_risk,
        critical_zones    = critical_zones,
        migration_paths   = migration_paths,
        quality_score     = quality
    )

    # ── Step 9: Save outputs ───────────────────────────────────
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{artefact_id}_salt"

        cv2.imwrite(str(save_dir / f"{prefix}_risk_map.png"),
                    render_risk_map_colour(risk_map))
        cv2.imwrite(str(save_dir / f"{prefix}_fluorescence.png"),
                    (fluorescence_map * 255).astype(np.uint8))
        cv2.imwrite(str(save_dir / f"{prefix}_composite.png"),
                    render_salt_composite(result))

        logger.info(f"Saved salt map outputs to: {save_dir}")

    return result


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — CALIBRATION
# ═══════════════════════════════════════════════════════════════

def calibrate_fluorescence_threshold(
    clean_sample_dir:  Path,
    salted_sample_dir: Path
) -> float:
    """
    Find the optimal fluorescence threshold for your UV LED setup.

    HOW TO USE:
        1. Scan 3–5 artefacts you know are clean (no salt)
           Save uv.png images to clean_sample_dir/sample_N/
        2. Sprinkle a tiny amount of table salt on a test stone
           Scan it and save to salted_sample_dir/sample_N/
        3. Run this function once
        4. Update FLUOR_THRESHOLD at top of this file

    Args:
        clean_sample_dir:  Directory of clean artefact UV scans
        salted_sample_dir: Directory of salted surface UV scans

    Returns:
        optimal_threshold: float
    """
    logger.info("Calibrating fluorescence threshold...")

    clean_means  = []
    salted_means = []

    for sample_dir, means, label in [
        (clean_sample_dir,  clean_means,  "clean"),
        (salted_sample_dir, salted_means, "salted")
    ]:
        for sub in sorted(Path(sample_dir).iterdir()):
            uv_path = sub / "uv.png"
            if not uv_path.exists():
                continue
            uv = cv2.imread(str(uv_path), cv2.IMREAD_GRAYSCALE)
            if uv is None:
                continue
            uv_f    = uv.astype(np.float32) / 255.0
            fluor   = extract_fluorescence_map(uv_f)
            means.append(float(fluor.mean()))
            logger.debug(f"  {label}/{sub.name}: mean={means[-1]:.3f}")

    if not clean_means or not salted_means:
        logger.error("Not enough samples")
        return FLUOR_THRESHOLD

    threshold = float((np.mean(clean_means) + np.mean(salted_means)) / 2)
    logger.info(f"Clean mean:  {np.mean(clean_means):.3f}")
    logger.info(f"Salted mean: {np.mean(salted_means):.3f}")
    logger.info(f"Recommended FLUOR_THRESHOLD = {threshold:.3f}")

    return threshold


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — STANDALONE TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Test the salt mapper without hardware.
    Generates synthetic UV image with simulated salt deposits.

    Run: python src/imaging/salt_mapper.py
    """
    print("AXUM — Salt Mapper Test (synthetic data)\n")

    h, w = 480, 640
    rng  = np.random.default_rng(42)

    # Synthetic visible: grey stone
    visible_syn = np.ones((h, w, 3), dtype=np.uint8) * 130
    noise       = rng.integers(0, 25, (h, w, 3), dtype=np.uint8)
    visible_syn = np.clip(visible_syn.astype(int) + noise, 0, 255)\
                    .astype(np.uint8)

    # Synthetic UV: low background + salt deposit zones
    uv_syn = np.ones((h, w), dtype=np.float32) * 0.15
    uv_syn += rng.normal(0, 0.02, (h, w)).astype(np.float32)

    # Salt deposit 1: large zone top-left
    Y, X = np.ogrid[:h, :w]
    zone1 = ((X - 150)**2 / 70**2 + (Y - 120)**2 / 40**2) <= 1
    uv_syn[zone1] += 0.55

    # Salt deposit 2: along a crack line (diagonal)
    for px in range(200, 450):
        py = int(h * 0.6 + (px - 200) * 0.3)
        if 0 <= py < h:
            uv_syn[max(0,py-3):py+3, px] += 0.45

    # Salt deposit 3: critical zone near centre
    zone3 = ((X - 400)**2 / 30**2 + (Y - 300)**2 / 25**2) <= 1
    uv_syn[zone3] += 0.80

    uv_syn = np.clip(uv_syn, 0, 1)

    # Synthetic inscription mask (horizontal band)
    insc_mask = np.zeros((h, w), dtype=np.uint8)
    insc_mask[280:320, 100:540] = 255

    # Synthetic conductivity grid (3×3)
    conduct_grid = [
        (100, 100, 800),  (320, 100, 1200), (540, 100, 400),
        (100, 300, 2800), (320, 300, 3600), (540, 300, 1800),
        (100, 400, 600),  (320, 400, 900),  (540, 400, 300),
    ]

    print("Running salt mapper on synthetic data...")
    result = run_salt_mapper(
        visible_image     = visible_syn,
        uv_image          = uv_syn,
        conductivity_grid = conduct_grid,
        inscription_mask  = insc_mask,
        save_dir          = Path("data/test_salt_mapper"),
        artefact_id       = "synthetic_test"
    )

    print(f"\nResults:")
    print(f"  Salt zones detected: {len(result.salt_zones)}")
    print(f"  Critical zones:      {len(result.critical_zones)}")
    print(f"  Overall risk:        {result.overall_risk:.1%}")
    print(f"  Migration paths:     {len(result.migration_paths)}")

    if result.salt_zones:
        z = result.salt_zones[0]
        print(f"\n  Worst zone:")
        print(f"    Risk level:       {z.risk_level} "
              f"({RISK_LABELS[z.risk_level]})")
        print(f"    Fluorescence:     {z.fluorescence:.2f}")
        print(f"    Conductivity:     {z.conductivity:.2f}")
        print(f"    Near inscription: {z.is_near_inscription}")
        print(f"    Area:             {z.area_px} px")

    print(f"\nSaved to: data/test_salt_mapper/")
    print("Open synthetic_test_salt_composite.png to view results")