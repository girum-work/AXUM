"""
AXUM ROVER — Hardware Control Package
======================================
WHAT:  Python interface to Arduino Mega (motors, arm, turntable, LEDs, sensors).
WHY:   All robot motion goes through serial commands — never raw serial.write() elsewhere.
STATUS: controller.py implements the serial transport, arm, turntable, and
camera interfaces. Hardware calls are safe to import but require the Arduino
and Pi endpoints in config.py to be configured before a live mission.

Exports:
    ArduinoSerial      — send_command("PING"), retry, port auto-detect
    ArmController      — POSE:PICK, GRIP, ARM angle commands
    TurntableController — STEP/ROTATE + PHOTO sequence for 36-shot scan
    CameraInterface    — HTTP capture from ESP32-CAM (config.ESP32_CAM_URL)

See docs/TECHNICAL_HANDOFF.md §7 and docs/CODEBASE_WALKTHROUGH.md §3.
"""

from src.arm.controller import (
    ArduinoSerial,
    ArmController,
    TurntableController,
    CameraInterface,
    MeshroomInterface
)

__all__ = [
    "ArduinoSerial",
    "ArmController",
    "TurntableController",
    "CameraInterface",
    "MeshroomInterface"
]
