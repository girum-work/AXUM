# AXUM — Serial Protocol Reference

**Maintained by:** Robotics Software Engineer, current as of Systems Integration Engineer's protocol reconciliation. This is the one document both `axum_rover.ino` and `controller.py` should be built against — if either side changes, update this file in the same change, not after.

**Wire basics:** 115200 baud, `Serial` (USB). Every command is one line, terminated `\n`. Every response is `OK:<...>` or `ERR:<...>`, one line, unless noted otherwise below.

---

## Connection

| Command | Response | Notes |
|---|---|---|
| `PING` | `PONG` | Used for serial port auto-discovery in `ArduinoSerial.connect()`. Only command without an `OK:`/`ERR:` prefix. |

## Drive

| Command | Response | Notes |
|---|---|---|
| `DRIVE:<left>,<right>` | `OK:DRIVE` | Both values -255..255, sign = direction. BTS7960 underneath; this interface is unchanged from the earlier L298N implementation by design — driver swap was contained entirely to firmware. |
| `STOP` | `OK:STOP` | Immediate. `DriveController.stop()` bypasses its own watchdog lock to call this without delay. |

## Arm

| Command | Response | Notes |
|---|---|---|
| `POSE:<name>` | `OK:POSE` | Named poses: `PARK`, `HOVER_TRAY`, `GRIP_TRAY`, `LIFT_CLEAR`, `HOVER_TABLE`, `PLACE_TABLE`, `LIFT_TABLE`. Foundation of `ArmController.go_pose()`, `pick_from_tray()`, `place_on_turntable()`, `park()`. |
| `ARM:<s0>,<s1>,<s2>,<s3>` | `OK:ARM` | All four joints, one call, degrees. |
| `GRIP:PULL` | `OK:GRIP:PULL` | Syringe actuator, calibrated pull position. Replaces the earlier `GRIP:<angle>` servo-claw command and the never-implemented `VACUUM:ON/OFF`. |
| `GRIP:RELEASE` | `OK:GRIP:RELEASE` | Syringe actuator, calibrated release position. |
| `CAMERA_TILT:<deg>` | `OK:CAMERA_TILT` | Rear/tail camera arm's tilt joint. Lives on `ArmController` (`camera_tilt()`), not `TurntableController` — this is the CTO's 2-DOF tail-arm ruling, unrelated to the navigation/turntable design. |

**Grip confidence — known gap, not yet closed:** nothing above reports whether a grip actually succeeded. `mission_tree.py`'s `PICK` phase currently uses a fabricated placeholder (`0.75`), which is why wiring the mission tree into `main_pipeline.py`'s default execution path is still held. `STATUS`'s `loadcell_raw` field is a real candidate for closing this gap if the load cell is physically positioned to sense grip weight — unconfirmed, not wired in yet.

## Turntable

| Command | Response | Notes |
|---|---|---|
| `STEP:<n>` | `OK:STEP` | Relative stepper motion. |
| `ROTATE:<deg>` | `OK:ROTATE` | Relative rotation in degrees. |
| `PHOTO` | `OK:PHOTO` | Trigger a capture. Internally moved from a GPIO pulse to `Serial1` (undercarriage cam) this revision — external shape unchanged, no `controller.py` change needed for this. |

`TurntableController.capture_rotation_set()` is built on `STEP`/`ROTATE`/`PHOTO`. There is no `capture_multi_angle_set()` and no tilt-sweep capability — `CAMERA_TILT` exists but is on the tail arm, not the turntable, so multi-angle photogrammetry scanning is architecturally a tail-arm behavior if it's ever built, not a turntable one. Worth confirming that's the intended design before anyone builds it.

## Lighting

| Command | Response | Notes |
|---|---|---|
| `LED:QUAD:<N\|E\|S\|W>` | `OK:LED:QUAD:<dir>` | One quadrant on at a time (photometric stereo). |
| `LED:UV:ON` / `LED:UV:OFF` | `OK:LED:UV:ON` / `OK:LED:UV:OFF` | Single-channel — UV fluorescence doesn't need directional lighting the way photometric stereo does. |

`FIRMWARE_LED_READY` in `config.py` gates whether advanced lighting captures
may be trusted. Flip it to `True` only after live bench verification. UV is
not an infrared source, so it must never be substituted for the IR frame
required by `multispectral.py`.

## Sensors

| Command | Response | Notes |
|---|---|---|
| `STATUS` | `{"front": <float cm>, "side": <float cm>, "ir_1": <int>, "ir_2": <int>, "ir_3": <int>, "loadcell_raw": <int>}` | JSON, no `OK:` prefix — a data query, not an action. **No battery voltage field.** No encoder fields — encoders are permanently removed from this project (Master's decision), not a temporary gap. `ir_1`-`3` purpose currently unconfirmed — asked, not yet answered. |
| `GPS_STATUS` | `{"fix": <bool>, "lat": <float>, "lon": <float>, "sats": <int>}` | JSON, matches `STATUS`'s pattern. Polled on request only — firmware does not push GPS data unprompted. NEO-7M updates at ~1Hz; polling faster doesn't get fresher data. GPS parsed continuously in the background via TinyGPS++ on `Serial3`; raw NMEA bytes are never written back to `Serial` under any code path — this was a deliberate design decision after a specific risk flag, not an oversight to "simplify" later. |
| `ATTITUDE` | `{"ok": <bool>, "roll": <float deg>, "pitch": <float deg>, "yaw": <float deg>, "biased": <bool>, "hz": <float>}` | JSON, same no-prefix data-query pattern as `STATUS`. GY-80 (ADXL345 + L3G4200D + HMC5883L), fused in firmware with a Mahony filter at 100Hz. `roll`/`pitch` are ±180/±90; `yaw` is **0..360**, not signed — the dashboard's compass rose depends on that. `ok: false` means no IMU was detected on the I2C bus at boot; every angle will read 0.00 and should be shown as "no signal", not as level. `biased: false` means the gyro zero-rate offset was never measured, so yaw will creep. Polled, not pushed — the firmware integrates continuously in `loop()` but only reports when asked. |
| `IMU:CALIBRATE` | `OK:IMU:CALIBRATE` / `ERR:IMU_ABSENT` | Re-measures gyro zero-rate offset by averaging 200 samples at rest. **Blocks the firmware for ~200 ms** — do not call it inside a drive loop. The rover must be stationary; firmware cannot detect that it isn't, so calibrating while moving bakes the motion into the bias. Also run once automatically at boot. |

The GY-80 supersedes the MPU6050 listed as unconfirmed in the firmware's integration note 7. It is three separate dies with no on-chip fusion and no calibration-status register, which is why orientation is computed on the Mega rather than read out of the sensor. Mahony was chosen over Madgwick because its correction step is a cross-product plus a PI term rather than a gradient-descent step with an inverse square root — on an FPU-less ATmega2560 that is what keeps the filter at 100Hz.

`mission_tree.py`'s supervisor uses `front`/`side` for a collision guard. The wheel-stall detector it also contains is permanently dormant — it handles missing encoder fields safely (`None`, not a crash) but has no sensor to actually check against anymore.

## Consolidant injection

| Command | Response | Notes |
|---|---|---|
| `INJECT:<...>` | `NACK:INJECT:MECHANISM_UNDECIDED` | **Deliberately not implemented.** Actuator mechanism (servo / stepper / DC motor) is an open Mechanical Engineer decision — the command's real shape depends entirely on which one is chosen. Do not implement against a guess. |

## Cameras (not on this serial link)

| Camera | Transport | Notes |
|---|---|---|
| Front nav webcam | USB → Raspberry Pi | Not on Arduino serial at all. Pi runs ArUco detection locally (Option A architecture), sends pose data to laptop over WiFi — separate protocol, not yet built. |
| Undercarriage ESP32-CAM | WiFi, relayed via `Serial1` internally for `PHOTO` triggering | Own network identity (Electronics' item). |
| Arm-angle ESP32-CAM | WiFi, `Serial2` | Triggered via `CAM_ARM_TRIGGER` — new this revision, not yet consumed by any `controller.py` code. |
| Pi IR-cut scan camera | Flask server (`pi_camera_server.py`), `PI_CAM_URL`/`PI_CAM_CAPTURE` | Primary scan camera. `CameraInterface` must be passed this URL explicitly — it does not default to it. |

---

## Known open items, tracked here so they're visible in one place
- `CAM_ARM_TRIGGER` exists in firmware and now on `ArduinoSerial.cam_arm_trigger()`, but nothing in `main_pipeline.py`/`mission_tree.py` calls it yet.
- `ir_1`/`ir_2`/`ir_3` purpose unconfirmed.
- `loadcell_raw` as a real grip-confidence source: unconfirmed whether the load cell is physically positioned for this.
- `INJECT` blocked on Mechanical's actuator decision.
- Wheel-stall detection has no sensor to check against; either remove the dead code path or find a replacement sensing source.
- **Added by Systems Integration Engineer, this pass:** `mission_tree.py`'s `_make_action_analyze` had three real bugs against the actual analysis-module signatures (wrong function names/kwargs for photometric stereo and multispectral, and a `stress_score` field that doesn't exist on `MultispectralResult`) — fixed, but the fix for multispectral's `visible`/`ir` mapping against the blackboard's `uv_path`/`nir_path` keys is a best guess, not confirmed, and depends on `_scan_artefact()` — which doesn't exist in `main_pipeline.py` yet — to actually define what gets saved under each key. `_ensure_hardware()` is also referenced but doesn't exist yet. Neither invented here; `run_artefact_mission()` stays unwired from `main_pipeline.py`'s default path until both exist and the mapping is confirmed.
