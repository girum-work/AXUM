"""
AXUM ROVER - Live demo control panel.

WHAT: A standalone Tkinter app for manually triggering robot behaviors in
front of judges — press a button, watch the robot do the thing.
WHY standalone instead of a dashboard panel: isolating this from the
mission dashboard means it has its own process, its own failure domain,
and no dependency on Flask/SocketIO being up. Fewer moving parts on stage.

SAFETY / INTERLOCK DESIGN — read before changing:
  - Only one process can hold the Arduino's serial port open at a time
    (pyserial raises SerialException on a busy port). That's the actual
    interlock against running this at the same time as main_pipeline.py's
    mission — it's enforced by the OS, not by a flag either app has to
    remember to check. If this app can't connect, the most likely reason
    is a mission is already running. Don't build a second, parallel lock
    that could drift out of sync with this one.
  - Drive buttons are hold-to-move: press sends repeated DRIVE commands
    every 150ms while held, release sends STOP immediately. DriveController
    also runs its own watchdog that auto-stops if commands stop refreshing
    for any reason (window losing focus, a hung callback, etc.) — this app
    does not rely solely on the button-release event.
  - Every command shows its real OK:/ERR: response in the log panel. No
    button silently "succeeds" — if firmware doesn't support a command
    yet (INJECT, as of this integration pass — VACUUM was the superseded
    design and never existed in firmware; CAMERA_TILT and GRIP:PULL/RELEASE
    are now real), the
    log shows the ERR, in red, immediately.
  - The STOP button is always enabled, always on top, and never queued
    behind other commands.

USAGE:
    python demo_control.py
"""

from __future__ import annotations

import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from src.arm.controller import ArduinoSerial, ArmController, DriveController, TurntableController

DRIVE_REPEAT_MS = 150
DRIVE_SPEED = 170
TURN_SPEED = 140
POSES = ["PARK", "HOVER_TRAY", "GRIP_TRAY", "LIFT_CLEAR", "HOVER_TABLE", "PLACE_TABLE", "LIFT_TABLE"]


class DemoControlApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AXUM — Demo Control")
        self.root.geometry("640x640")

        self.arduino: ArduinoSerial | None = None
        self.arm: ArmController | None = None
        self.drive: DriveController | None = None
        self.turntable: TurntableController | None = None
        self._drive_repeat_job: str | None = None
        self._telemetry_job: str | None = None
        self.controls: list[tk.Button] = []

        self._build_ui()
        self._connect()

    # ── connection ──────────────────────────────────────────────

    def _connect(self) -> None:
        try:
            self.arduino = ArduinoSerial()
            self.arm = ArmController(self.arduino)
            self.drive = DriveController(self.arduino)
            self.turntable = TurntableController(arduino=self.arduino, camera=None)
            self._set_status(f"Connected on {self.arduino.port}", ok=True)
            self._set_controls_enabled(True)
            self._start_telemetry_polling()
        except Exception as exc:
            self._set_status(f"NOT CONNECTED: {exc}", ok=False)
            self._log(f"Connection failed: {exc}", error=True)
            self._log("If a mission is currently running, this is expected — "
                       "only one process can hold the serial port.", error=False)
            self._set_controls_enabled(False)

    # ── UI construction ─────────────────────────────────────────

    def _build_ui(self) -> None:
        self.status_label = tk.Label(self.root, text="Connecting...", font=("Arial", 12, "bold"))
        self.status_label.pack(pady=6)

        stop_btn = tk.Button(
            self.root, text="■ STOP", font=("Arial", 16, "bold"),
            bg="#c0392b", fg="white", height=2, command=self._on_stop,
        )
        stop_btn.pack(fill="x", padx=10, pady=6)

        self.reconnect_btn = tk.Button(self.root, text="Reconnect", command=self._connect)
        self.reconnect_btn.pack(pady=2)

        self._section("Drive")
        drive_frame = tk.Frame(self.root)
        drive_frame.pack(pady=4)
        self._hold_button(drive_frame, "▲ Forward", 0, 1, self._forward)
        self._hold_button(drive_frame, "◀ Left", 1, 0, self._turn_left)
        self._hold_button(drive_frame, "STOP", 1, 1, self._on_stop, hold=False, bg="#c0392b", fg="white")
        self._hold_button(drive_frame, "Right ▶", 1, 2, self._turn_right)
        self._hold_button(drive_frame, "▼ Back", 2, 1, self._backward)

        self._section("Arm — poses")
        pose_frame = tk.Frame(self.root)
        pose_frame.pack(pady=4)
        for i, pose in enumerate(POSES):
            b = tk.Button(pose_frame, text=pose, width=12,
                          command=lambda p=pose: self._run(lambda: self.arm.go_pose(p), f"POSE:{p}"))
            b.grid(row=i // 4, column=i % 4, padx=3, pady=3)
            self.controls.append(b)

        self._section("Gripper")
        grip_frame = tk.Frame(self.root)
        grip_frame.pack(pady=4)
        # ArmController.grip_pull()/grip_release() now exist and match real
        # firmware's GRIP:PULL/GRIP:RELEASE commands (added + verified in
        # this integration pass — was blocked, mechanism was undecided when
        # this file was originally written). Angles behind these commands
        # are still PROVISIONAL pending Mechanical's crank throw numbers —
        # that's a firmware-side calibration concern, not a reason to keep
        # this UI disabled.
        self._btn(grip_frame, "Grip (pull)", lambda: self._run(self.arm.grip_pull, "GRIP:PULL")).grid(row=0, column=0, padx=4)
        self._btn(grip_frame, "Release", lambda: self._run(self.arm.grip_release, "GRIP:RELEASE")).grid(row=0, column=1, padx=4)

        self._section("Demo macros")
        macro_frame = tk.Frame(self.root)
        macro_frame.pack(pady=4)
        self._btn(macro_frame, "Pick nearby object", self._demo_pick).grid(row=0, column=0, padx=4)
        self._btn(macro_frame, "Place on turntable", lambda: self._run(self.arm.place_on_turntable, "PLACE_SEQ")).grid(row=0, column=1, padx=4)
        self._btn(macro_frame, "Park arm", lambda: self._run(self.arm.park, "PARK")).grid(row=0, column=2, padx=4)

        self._section("Turntable / camera arm")
        tt_frame = tk.Frame(self.root)
        tt_frame.pack(pady=4)
        self._btn(tt_frame, "Rotate -90°", lambda: self._run(lambda: self.turntable.rotate_degrees(-90), "ROTATE")).grid(row=0, column=0, padx=4)
        self._btn(tt_frame, "Rotate +90°", lambda: self._run(lambda: self.turntable.rotate_degrees(90), "ROTATE")).grid(row=0, column=1, padx=4)
        # ArmController.camera_tilt(angle) now exists and matches real
        # firmware's CAMERA_TILT command (added + verified in this
        # integration pass — was blocked when this file was originally
        # written). Note it's on ArmController, not TurntableController.
        self._btn(tt_frame, "Camera tilt 0°", lambda: self._run(lambda: self.arm.camera_tilt(0), "CAMERA_TILT:0")).grid(row=0, column=2, padx=4)
        self._btn(tt_frame, "Camera tilt 45°", lambda: self._run(lambda: self.arm.camera_tilt(45), "CAMERA_TILT:45")).grid(row=0, column=3, padx=4)

        self._section("Intervention (requires firmware patch)")
        inject_frame = tk.Frame(self.root)
        inject_frame.pack(pady=4)
        # inject_consolidant() does not exist on ArmController — same
        # treatment, disabled rather than silently broken.
        self._blocked_btn(inject_frame, "Inject consolidant (no method/firmware yet)", 0, 0)

        self._section("Live telemetry (STATUS)")
        telemetry_frame = tk.Frame(self.root)
        telemetry_frame.pack(pady=4)
        # Fields shown here are exactly what axum_rover.ino's STATUS handler
        # returns — confirmed by reading the firmware directly, not guessed.
        # enc_l/enc_r REMOVED this pass -- encoders are no longer in the
        # build (pins 18/19 now belong to Serial1, undercarriage ESP32-CAM).
        # ir_1/ir_2/ir_3/loadcell_raw ADDED -- loadcell_raw is genuinely
        # uncalibrated (no set_scale() call in firmware yet), shown as
        # raw counts, not grams -- do not relabel it without a matching
        # firmware calibration pass, or this becomes exactly the kind of
        # silently-wrong display this app is built to avoid.
        self.telemetry_labels: dict[str, tk.Label] = {}
        for i, field in enumerate(["front", "side", "ir_1", "ir_2", "ir_3", "loadcell_raw"]):
            tk.Label(telemetry_frame, text=field, font=("Consolas", 9, "bold")).grid(row=0, column=i, padx=6)
            value_label = tk.Label(telemetry_frame, text="—", font=("Consolas", 11))
            value_label.grid(row=1, column=i, padx=6)
            self.telemetry_labels[field] = value_label

        self._section("Log")
        self.log_text = tk.Text(self.root, height=10, font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True, padx=10, pady=4)
        self.log_text.tag_config("error", foreground="#c0392b")
        self.log_text.tag_config("ok", foreground="#27ae60")

    def _section(self, title: str) -> None:
        tk.Label(self.root, text=title, font=("Arial", 10, "bold")).pack(pady=(8, 0))

    def _btn(self, parent, text, command) -> tk.Button:
        b = tk.Button(parent, text=text, command=command)
        self.controls.append(b)
        return b

    def _blocked_btn(self, parent, text: str, row: int, col: int) -> tk.Button:
        """
        A button for functionality that isn't real yet — either the
        controller.py method doesn't exist, the firmware command doesn't
        exist, or both. Deliberately NOT added to self.controls, so
        _set_controls_enabled(True) on connect can never accidentally
        re-enable it and make it look functional. The label states why
        it's disabled rather than just being greyed out with no context —
        a demo panel with an unexplained dead button is worse than one
        that's honest about what's missing.
        """
        b = tk.Button(parent, text=text, state="disabled", fg="#999999")
        b.grid(row=row, column=col, padx=4)
        return b

    def _hold_button(self, parent, text, row, col, action, hold: bool = True, bg=None, fg=None):
        kwargs = {"text": text, "width": 10}
        if bg:
            kwargs["bg"] = bg
        if fg:
            kwargs["fg"] = fg
        b = tk.Button(parent, **kwargs)
        b.grid(row=row, column=col, padx=3, pady=3)
        if hold:
            b.bind("<ButtonPress-1>", lambda e: self._start_drive_repeat(action))
            b.bind("<ButtonRelease-1>", lambda e: self._stop_drive_repeat())
        else:
            b.config(command=action)
        self.controls.append(b)
        return b

    # ── drive hold-to-move ──────────────────────────────────────

    def _start_drive_repeat(self, action) -> None:
        self._stop_drive_repeat()

        def tick():
            self._run(action, "DRIVE", quiet=True)
            self._drive_repeat_job = self.root.after(DRIVE_REPEAT_MS, tick)

        tick()

    def _stop_drive_repeat(self) -> None:
        if self._drive_repeat_job:
            self.root.after_cancel(self._drive_repeat_job)
            self._drive_repeat_job = None
        self._on_stop()

    # ── live telemetry polling ───────────────────────────────────

    TELEMETRY_POLL_MS = 300  # ~3Hz — enough to look "live" without spamming the serial link

    def _start_telemetry_polling(self) -> None:
        if self._telemetry_job is not None:
            self.root.after_cancel(self._telemetry_job)
            self._telemetry_job = None
        self._poll_telemetry_once()

    def _poll_telemetry_once(self) -> None:
        if self.arduino is not None:
            try:
                status_payload = self.arduino.status()
                if isinstance(status_payload, dict):
                    for field, label in self.telemetry_labels.items():
                        value = status_payload.get(field)
                        label.config(text="—" if value is None else str(value))
            except Exception as exc:
                # Don't spam the log panel for a routine background poll —
                # command buttons already show errors loudly; a telemetry
                # hiccup shouldn't drown that signal out. Blank the display
                # instead so it's visibly "not updating" rather than
                # silently showing a stale number.
                for label in self.telemetry_labels.values():
                    label.config(text="?")
                logger.debug(f"Telemetry poll failed: {exc}")
        self._telemetry_job = self.root.after(self.TELEMETRY_POLL_MS, self._poll_telemetry_once)

    def _forward(self):
        return self.drive.forward(DRIVE_SPEED)

    def _backward(self):
        return self.drive.backward(DRIVE_SPEED)

    def _turn_left(self):
        return self.drive.turn_left(TURN_SPEED)

    def _turn_right(self):
        return self.drive.turn_right(TURN_SPEED)

    def _on_stop(self) -> None:
        if self.drive is not None:
            self._run(self.drive.stop, "STOP", quiet=True)

    def _demo_pick(self) -> None:
        # NOTE: pick_from_tray() is the canned arm macro, not vision-guided
        # targeting — "nearby object" means "whatever's in the tray slot."
        self._run(self.arm.pick_from_tray, "PICK_SEQ")

    # ── command execution + logging ─────────────────────────────

    def _run(self, fn, label: str, quiet: bool = False) -> None:
        if self.arduino is None:
            self._log(f"{label}: not connected", error=True)
            return
        try:
            result = fn()
            if not quiet:
                self._log(f"{label} -> {result}", error=False)
        except Exception as exc:
            self._log(f"{label} -> ERR: {exc}", error=True)

    def _log(self, message: str, error: bool) -> None:
        timestamp = time.strftime("%H:%M:%S")
        tag = "error" if error else "ok"
        self.log_text.insert("end", f"[{timestamp}] {message}\n", tag)
        self.log_text.see("end")

    def _set_status(self, text: str, ok: bool) -> None:
        self.status_label.config(text=text, fg="#27ae60" if ok else "#c0392b")

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in self.controls:
            widget.config(state=state)


def main() -> None:
    root = tk.Tk()
    DemoControlApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()