# AXUM Firmware Pin Map

Authoritative source: `arduino/at_mega/axum_rover.ino`. Update this file in
the same change whenever firmware pin assignments change.

| Subsystem | Mega pins | Notes |
|---|---|---|
| Left BTS7960 | D7 RPWM, D8 LPWM, D44 enable | Differential drive |
| Right BTS7960 | D12 RPWM, D13 LPWM, D45 enable | Differential drive |
| Arm servos | D3, D4, D6, D9, D10 | S0–S4; S4 is gripper actuator |
| Tail camera tilt | D46 | Tilt only; no reach servo implemented |
| Turntable A4988 | D22 STEP, D23 DIR, D24 enable | Enable is active-low |
| Front ultrasonic | D26 trigger, D27 echo | STATUS `front` |
| Side ultrasonic | D28 trigger, D29 echo | STATUS `side` |
| HX711 load cell | D31 DT, D32 SCK | STATUS `loadcell_raw` |
| IR sensors | A0, A1, A2 | STATUS `ir_1`–`ir_3` |
| GPS | Serial3 | NEO-7M parsed by TinyGPS++ |
| Under-carriage ESP32-CAM | Serial1 | `PHOTO` trigger |
| Arm-angle ESP32-CAM | Serial2 | `CAM_ARM_TRIGGER` |

## Reserved / unassigned

A3 and A4 are unassigned in firmware. Battery-voltage sensing must not use
either until the divider resistor values, battery voltage range, and ADC
calibration procedure are approved.
