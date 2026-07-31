/*
 * AXUM ROVER — Main Arduino Firmware
 * ====================================
 * Controls:
 *   - 4x drive motors via 2x BTS7960 high-current H-bridge drivers
 *   - 4x arm servos (base, shoulder, elbow, wrist) + 1x gripper servo
 *   - 1x rear-arm camera tilt servo
 *   - 1x NEMA17 stepper (turntable) via A4988 STEP/DIR driver
 *   - 4x quadrant LEDs (N/E/S/W, photometric stereo) + 1x UV LED,
 *     all driven through a PCA9685 PWM driver into MOSFETs
 *   - HC-SR04 ultrasonic sensors (front + side)
 *   - 1x encoder per side (wheel odometry)
 *
 * Communication: USB Serial at 115200 baud
 * Protocol: newline-terminated ASCII commands
 *
 * Command reference:
 *   DRIVE:<left_speed>,<right_speed>   speed: -255 to 255 (UNCHANGED shape)
 *   ARM:<s0>,<s1>,<s2>,<s3>            angles: 0 to 180 degrees (UNCHANGED)
 *   GRIP:PULL                          syringe-actuated vacuum grip: engage
 *   GRIP:RELEASE                       syringe-actuated vacuum grip: release
 *   STEP:<steps>                        turntable steps (+ = CW) (UNCHANGED)
 *   ROTATE:<degrees>                    turntable rotation (UNCHANGED)
 *   CAMERA_TILT:<deg>                   rear-arm camera tilt servo, 0-180
 *   LED:QUAD:<N|E|S|W>                  fire one photometric-stereo quadrant
 *                                        (others off); N/E/S/W only
 *   LED:QUAD:OFF                        all quadrant LEDs off
 *   LED:UV:ON / LED:UV:OFF              fluorescence UV LED (independent
 *                                        of quadrant LEDs — no directional
 *                                        requirement)
 *   PHOTO                               trigger ESP32-CAM (nav/fallback)
 *                                        capture pulse — see integration
 *                                        note below re: Pi camera path
 *   PING                                returns PONG (connection check)
 *   GPS_STATUS                          {"fix":bool,"lat":...,"lon":...,"sats":...}
 *   CAM_ARM_TRIGGER                     trigger arm-angle ESP32-CAM (Serial2)
 *   STOP                                emergency stop all motion
 *   STATUS                              returns sensor readings JSON
 *
 * ─────────────────────────────────────────────────────────────────
 * INTEGRATION NOTES (Systems Integration Engineer, this pass)
 * ─────────────────────────────────────────────────────────────────
 * 1. ENCODERS: REMOVED from this build (confirmed by Master this pass --
 *    "we will NOT be using any rotary encoders for now, that's why the
 *    ESP32 cameras will be wired there instead"). This resolves a real
 *    conflict flagged across the last two integration passes: Serial1 is
 *    hardware-fixed to pins 18/19 on the Mega (not reassignable like a
 *    normal digital pin) and cannot coexist with encoder ISRs on those
 *    same pins. Pins 18/19 now belong to Serial1 (undercarriage
 *    ESP32-CAM, see note 6). STATUS no longer reports enc_l/enc_r --
 *    if this breaks anything reading those fields (demo_control.py's
 *    telemetry panel did), that's been updated in the same pass. If
 *    encoders come back into scope later, they need different pins.
 *
 * 2. INJECT command: NOT implemented. No actuator command spec has been
 *    supplied by Electronics yet (angle pattern, calibration constants).
 *    Do not guess this — implementing a wrong angle pattern against a
 *    real consolidant syringe is a physical-damage risk, not just a bug.
 *
 * 3. How the Pi 4 IR-CUT camera's capture is actually triggered — a second
 *    Arduino output pin into the Pi's GPIO, or purely an HTTP call from
 *    the laptop with no Arduino involvement at all — was not specified
 *    anywhere I could find. Not invented here; flagging as an open
 *    integration question rather than guessing at wiring that could be
 *    physically wrong.
 *
 * 4. GRIP_PULL_ANGLE / GRIP_RELEASE_ANGLE below are PROVISIONAL. Mechanical
 *    has not supplied final crank throw/force numbers for the syringe
 *    linkage. Values chosen to match the existing pick/place pose
 *    convention already used elsewhere in this file (90 = grip-closed
 *    posture, 160 = open posture) purely for continuity — verify against
 *    the real mechanism before trusting these on hardware.
 *
 * 5. New library dependencies, this pass: PCA9685 LED driving requires
 *    Adafruit_PWMServoDriver (+ Adafruit_BusIO). GPS requires TinyGPS++
 *    (Mikal Hart). Load cell requires HX711 (bogde). None of these were
 *    previously used in this firmware — all need installing before this
 *    compiles.
 *
 * 6. THIS PASS incorporates GPS, ESP32-CAM support, 3x IR, and the load
 *    cell from a collected draft — deliberately EXCLUDING that draft's
 *    FSR sensors, piezo, and MPU6050 (no BOM/checkpoint confirmation seen
 *    for those; not wiring undocumented hardware in). Specifics:
 *
 *    - GPS (NEO-7M, Serial3, pins 14/15): kept exactly as designed in the
 *      collected draft -- genuinely good design. Parsed continuously in
 *      the background via TinyGPS++, exposed only on request via
 *      GPS_STATUS. Raw Serial3 bytes are NEVER echoed to Serial (the
 *      command channel) -- there is no code path where a GPS NMEA byte
 *      can land inside a command response. Do not "simplify" this into a
 *      raw relay later.
 *
 *    - ESP32-CAM, arm-angle (Serial2, pins 16/17): genuinely free, no
 *      conflict, wired as real hardware UART. Mega sends a short text
 *      trigger ("CAPTURE\n") on request via CAM_ARM_TRIGGER; it does NOT
 *      relay image bytes back over Serial2->Serial, for the same reason
 *      GPS doesn't raw-relay -- JPEG frames over a shared low-bandwidth
 *      UART would be slow and would risk corrupting the command channel.
 *      The actual JPEG still needs to be fetched over WiFi HTTP by
 *      whatever's consuming it (laptop), same pattern as the existing
 *      camera. The ESP32-CAM's own firmware (separate .ino, out of scope
 *      here) needs to listen for "CAPTURE" on its UART and serve the
 *      resulting frame over HTTP as before.
 *
 *    - ESP32-CAM, undercarriage/nav (Serial1, pins 18/19): now wired as
 *      real hardware UART, same non-relay trigger-only pattern as the
 *      arm-angle camera. PHOTO command migrated from its old GPIO-pulse
 *      implementation to Serial1.print("CAPTURE\n") -- external command
 *      shape unchanged (still PHOTO -> OK:PHOTO), so no Python-side
 *      change was needed for this one.
 *
 *    - 3x IR sensors (A0/A1/A2): simple analogRead, folded into STATUS's
 *      existing JSON response as ir_1/ir_2/ir_3 rather than a new command
 *      -- ArduinoSerial.status() on the Python side already parses
 *      whatever JSON keys STATUS returns, so this is additive and doesn't
 *      touch any existing consumer.
 *
 *    - Load cell (HX711, pins 31/32 -- NOT the collected draft's 30, which
 *      collides with this file's existing CAM_TRIGGER): also folded into
 *      STATUS as loadcell_raw (uncalibrated -- no set_scale() call yet,
 *      not real grams), reporting null when the amp isn't ready rather
 *      than blocking the STATUS response.
 *
 *    NOT included, explicitly excluded per this request: 2x FSR sensors,
 *    piezo, MPU6050 IMU. No BOM/checkpoint confirmation seen for any of
 *    these -- ask again with that confirmation if they're real.
 */

#include <Servo.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <TinyGPS++.h>
#include <HX711.h>

// ═══════════════════════════════════════════════════════════════
// PIN DEFINITIONS
// ═══════════════════════════════════════════════════════════════

// Drive motors — BTS7960 #1 (LEFT side)
// RPWM/LPWM: PWM speed+direction. EN: R_EN and L_EN tied together on most
// BTS7960 breakout boards -- one enable pin per side is sufficient.
const int L_RPWM = 7;
const int L_LPWM = 8;
const int L_EN   = 44;

// Drive motors — BTS7960 #2 (RIGHT side)
const int R_RPWM = 12;
const int R_LPWM = 13;
const int R_EN   = 45;

// Arm servos
const int PIN_S0 = 3;    // Base rotation (MG996R)
const int PIN_S1 = 4;    // Shoulder       (MG996R)
const int PIN_S2 = 6;    // Elbow          (MG90S)
const int PIN_S3 = 9;    // Wrist tilt     (SG90)
const int PIN_S4 = 10;   // Gripper / syringe actuator (SG90)

// Rear-arm camera tilt (2-DOF rear arm: tilt + reach; azimuth is covered
// by turntable rotation. REACH joint command not yet specified — not
// implemented here, flagging rather than inventing.)
const int PIN_CAMERA_TILT = 46;

// Turntable stepper (NEMA17 via A4988)
const int TT_STEP   = 22;
const int TT_DIR     = 23;
const int TT_ENABLE  = 24;   // active LOW on A4988 -- LOW = driver enabled
// Pin 25 (unused 4th 28BYJ-48 coil pin) freed by this migration.

// PCA9685 LED driver (I2C — fixed SDA=20, SCL=21 on Mega, no pin consts
// needed). Channels 0-3 = quadrant LEDs (photometric stereo), channel 4 =
// UV LED (fluorescence detection — single unit is correct; no directional
// requirement unlike the quadrant LEDs' surface-normal math).
Adafruit_PWMServoDriver ledDriver = Adafruit_PWMServoDriver();
const uint8_t LED_CH_N  = 0;
const uint8_t LED_CH_E  = 1;
const uint8_t LED_CH_S  = 2;
const uint8_t LED_CH_W  = 3;
const uint8_t LED_CH_UV = 4;
const uint16_t LED_PWM_FULL_ON = 4095;
const uint16_t LED_PWM_OFF     = 0;

// Ultrasonic sensors
const int FRONT_TRIG = 26;
const int FRONT_ECHO = 27;
const int SIDE_TRIG  = 28;
const int SIDE_ECHO  = 29;

// ESP32-CAM trigger (nav/fallback camera; HIGH pulse triggers capture)
// Pin 30 freed by this migration (was CAM_TRIGGER's GPIO pulse pin --
// undercarriage camera now uses Serial1 instead, see below).

// Load cell (HX711) -- NOT pins 30/31 as in the collected draft (30
// collides with CAM_TRIGGER above); moved to 31/32, both genuinely free.
const int HX711_DT  = 31;
const int HX711_SCK = 32;

// IR sensors (3x, analog) -- genuinely free, no conflicts.
const int PIN_IR_1 = A0;
const int PIN_IR_2 = A1;
const int PIN_IR_3 = A2;

// Serial1 (hardware-fixed pins 18/19) -- ESP32-CAM, undercarriage/nav.
// Encoders were removed from this build (confirmed by Master this pass --
// "we will NOT be using any rotary encoders for now, that's why the ESP32
// cameras will be wired there instead"). This resolves the conflict
// flagged in the last two integration passes: Serial1 is hardware-fixed
// to pins 18/19 and cannot coexist with encoder ISRs on the same pins.
// If encoders come back into scope later, they need different pins --
// not 18/19, which now belong to Serial1.

// GPS (NEO-7M) -- Serial3, hardware-fixed to pins 14/15 on the Mega, no
// #define needed. Outdoor/transport-phase navigation only; indoor arena
// navigation still uses ArUco via the Pi, unrelated to this.

// ESP32-CAM, arm-angle -- Serial2, hardware-fixed to pins 16/17, genuinely
// free. See integration note #6 for why this one gets real UART and the
// undercarriage camera doesn't.

// ═══════════════════════════════════════════════════════════════
// OBJECTS
// ═══════════════════════════════════════════════════════════════

Servo s0, s1, s2, s3, s4, sCameraTilt;
TinyGPSPlus gps;
HX711 loadCell;

// NEMA17: 200 full steps/rev (1.8 deg/step). A4988 default is full-step
// (no microstepping jumpers set) -- adjust STEPS_PER_REV if MS1/MS2/MS3
// microstepping is wired. Exact per-phase current is set on the A4988's
// physical current-limit trimpot, not in this logic -- unconfirmed but
// doesn't block writing/testing the STEP/DIR sequencing itself.
const int STEPS_PER_REV = 200;

// Current arm pose [s0, s1, s2, s3, s4]
int currentPose[5] = {90, 30, 150, 90, 160};

// Gripper state — provisional angles, see integration note #4 above.
const int GRIP_PULL_ANGLE    = 90;   // engage vacuum grip
const int GRIP_RELEASE_ANGLE = 160;  // release

// Encoder counts
// Encoder globals removed -- encoders not in this build (see pin note above).

// ═══════════════════════════════════════════════════════════════
// PRE-COMPUTED POSES
// Format: {s0, s1, s2, s3, s4(gripper)}
// Calibrate these values on your actual physical arm.
// Use the calibration sketch (below) to find correct angles.
// ═══════════════════════════════════════════════════════════════

const int POSE_PARK[5]         = {90,  30, 150, 90, 160};  // folded safe
const int POSE_HOVER_TRAY[5]   = {45,  75, 110, 85, 160};  // above tray
const int POSE_GRIP_TRAY[5]    = {45,  90, 130, 80, 160};  // at object
const int POSE_LIFT_CLEAR[5]   = {45,  60, 100, 85, 90};   // lifted
const int POSE_HOVER_TABLE[5]  = {135, 70, 115, 85, 90};   // above turntable
const int POSE_PLACE_TABLE[5]  = {135, 85, 130, 80, 90};   // on turntable
const int POSE_LIFT_TABLE[5]   = {135, 60, 100, 85, 160};  // lifted from table

// ═══════════════════════════════════════════════════════════════
// SETUP
// ═══════════════════════════════════════════════════════════════

void setup() {
  Serial.begin(115200);
  Serial1.begin(115200);   // ESP32-CAM, undercarriage (real hardware UART --
                            // encoders removed this pass, pins 18/19 freed)
  Serial2.begin(115200);   // ESP32-CAM, arm-angle (real hardware UART)
  Serial3.begin(9600);     // GPS NEO-7M -- default baud, NMEA output

  // Attach servos
  s0.attach(PIN_S0);
  s1.attach(PIN_S1);
  s2.attach(PIN_S2);
  s3.attach(PIN_S3);
  s4.attach(PIN_S4);
  sCameraTilt.attach(PIN_CAMERA_TILT);
  sCameraTilt.write(90);  // neutral tilt on boot

  // Drive motor pins (BTS7960)
  pinMode(L_RPWM, OUTPUT); pinMode(L_LPWM, OUTPUT); pinMode(L_EN, OUTPUT);
  pinMode(R_RPWM, OUTPUT); pinMode(R_LPWM, OUTPUT); pinMode(R_EN, OUTPUT);
  digitalWrite(L_EN, HIGH);  // BTS7960 EN is active HIGH
  digitalWrite(R_EN, HIGH);

  // Turntable stepper pins (A4988)
  pinMode(TT_STEP, OUTPUT);
  pinMode(TT_DIR, OUTPUT);
  pinMode(TT_ENABLE, OUTPUT);
  digitalWrite(TT_ENABLE, HIGH);  // start disabled (active LOW) -- no
                                    // holding current until first move

  // LED driver (PCA9685)
  Wire.begin();
  ledDriver.begin();
  ledDriver.setPWMFreq(1000);
  allLedsOff();

  // Sensor pins
  pinMode(FRONT_TRIG, OUTPUT); pinMode(FRONT_ECHO, INPUT);
  pinMode(SIDE_TRIG,  OUTPUT); pinMode(SIDE_ECHO,  INPUT);

  // Load cell -- analog IR pins need no pinMode() call (analogRead only)
  loadCell.begin(HX711_DT, HX711_SCK);
  // NOTE: no loadCell.set_scale() call -- get_units() returns raw/1.0
  // (effectively uncalibrated counts) until a real calibration factor is
  // measured against known weights. STATUS's loadcell_raw field is not
  // real grams yet; treat it as a relative/uncalibrated reading.

  // No encoder interrupts -- not in this build (see pin note above).

  // Move to park pose on startup
  moveToPose(POSE_PARK, 3);
  s4.write(GRIP_RELEASE_ANGLE);
  currentPose[4] = GRIP_RELEASE_ANGLE;

  // Stop motors
  stopMotors();

  Serial.println("AXUM_READY");
}

// ═══════════════════════════════════════════════════════════════
// MAIN LOOP — Command Parser
// ═══════════════════════════════════════════════════════════════

void loop() {
  // GPS: feed the parser continuously in the background. This NEVER writes
  // to Serial (the command channel) -- only into TinyGPS++'s internal
  // state. GPS_STATUS below is the only way this data reaches the outside
  // world. Do not "simplify" this into a raw relay -- see integration
  // note #6.
  while (Serial3.available()) {
    gps.encode(Serial3.read());
  }

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    processCommand(cmd);
  }
}

void processCommand(String cmd) {
  // ── PING ─────────────────────────────────────────────────────
  if (cmd == "PING") {
    Serial.println("PONG");
    return;
  }

  // ── STOP ─────────────────────────────────────────────────────
  if (cmd == "STOP") {
    stopMotors();
    Serial.println("OK:STOP");
    return;
  }

  // ── STATUS ───────────────────────────────────────────────────
  if (cmd == "STATUS") {
    float frontDist = readUltrasonic(FRONT_TRIG, FRONT_ECHO);
    float sideDist  = readUltrasonic(SIDE_TRIG,  SIDE_ECHO);
    Serial.print("{\"front\":");  Serial.print(frontDist);
    Serial.print(",\"side\":");   Serial.print(sideDist);
    Serial.print(",\"ir_1\":");   Serial.print(analogRead(PIN_IR_1));
    Serial.print(",\"ir_2\":");   Serial.print(analogRead(PIN_IR_2));
    Serial.print(",\"ir_3\":");   Serial.print(analogRead(PIN_IR_3));
    Serial.print(",\"loadcell_raw\":");
    if (loadCell.is_ready()) {
      Serial.print(loadCell.get_units(3));
    } else {
      Serial.print("null");
    }
    Serial.println("}");
    return;
  }

  // ── DRIVE:<left>,<right> ──────────────────────────────────────
  if (cmd.startsWith("DRIVE:")) {
    String params = cmd.substring(6);
    int comma = params.indexOf(',');
    if (comma < 0) { Serial.println("ERR:DRIVE_SYNTAX"); return; }

    int leftSpeed  = params.substring(0, comma).toInt();
    int rightSpeed = params.substring(comma + 1).toInt();

    setMotors(leftSpeed, rightSpeed);
    Serial.println("OK:DRIVE");
    return;
  }

  // ── ARM:<s0>,<s1>,<s2>,<s3> ───────────────────────────────────
  if (cmd.startsWith("ARM:")) {
    String params = cmd.substring(4);
    int angles[4];
    int idx = 0;

    while (params.length() > 0 && idx < 4) {
      int comma = params.indexOf(',');
      if (comma < 0) {
        angles[idx++] = params.toInt();
        break;
      }
      angles[idx++] = params.substring(0, comma).toInt();
      params = params.substring(comma + 1);
    }

    if (idx < 4) { Serial.println("ERR:ARM_SYNTAX"); return; }

    // Move smoothly
    int targetPose[5] = {angles[0], angles[1], angles[2], angles[3],
                         currentPose[4]};  // preserve gripper
    moveToPose(targetPose, 2);
    Serial.println("OK:ARM");
    return;
  }

  // ── GRIP:PULL / GRIP:RELEASE ────────────────────────────────────
  // Syringe-actuated vacuum gripper. Named commands only -- angle is
  // never passed in from the caller (see integration note #4: the exact
  // angles are provisional pending Mechanical's crank throw numbers).
  if (cmd == "GRIP:PULL") {
    driveGripperTo(GRIP_PULL_ANGLE);
    Serial.println("OK:GRIP");
    return;
  }
  if (cmd == "GRIP:RELEASE") {
    driveGripperTo(GRIP_RELEASE_ANGLE);
    Serial.println("OK:GRIP");
    return;
  }
  if (cmd.startsWith("GRIP:")) {
    // Old numeric protocol (GRIP:<angle>) is retired -- reject rather than
    // silently accept, so a caller still on the old protocol gets a clear
    // error instead of the gripper silently moving to the wrong position.
    Serial.println("ERR:GRIP_PROTOCOL_RETIRED_USE_PULL_OR_RELEASE");
    return;
  }

  // ── POSE:<n> ───────────────────────────────────────────────
  if (cmd.startsWith("POSE:")) {
    String poseName = cmd.substring(5);
    if      (poseName == "PARK")         moveToPose(POSE_PARK, 3);
    else if (poseName == "HOVER_TRAY")   moveToPose(POSE_HOVER_TRAY, 2);
    else if (poseName == "GRIP_TRAY")    moveToPose(POSE_GRIP_TRAY, 2);
    else if (poseName == "LIFT_CLEAR")   moveToPose(POSE_LIFT_CLEAR, 2);
    else if (poseName == "HOVER_TABLE")  moveToPose(POSE_HOVER_TABLE, 2);
    else if (poseName == "PLACE_TABLE")  moveToPose(POSE_PLACE_TABLE, 2);
    else if (poseName == "LIFT_TABLE")   moveToPose(POSE_LIFT_TABLE, 2);
    else { Serial.println("ERR:UNKNOWN_POSE"); return; }
    Serial.println("OK:POSE");
    return;
  }

  // ── CAMERA_TILT:<deg> ────────────────────────────────────────
  if (cmd.startsWith("CAMERA_TILT:")) {
    int angle = cmd.substring(12).toInt();
    angle = constrain(angle, 0, 180);
    sCameraTilt.write(angle);
    Serial.println("OK:CAMERA_TILT");
    return;
  }

  // ── STEP:<steps> ──────────────────────────────────────────────
  if (cmd.startsWith("STEP:")) {
    int steps = cmd.substring(5).toInt();
    stepTurntable(steps);
    Serial.println("OK:STEP");
    return;
  }

  // ── ROTATE:<degrees> ──────────────────────────────────────────
  // Convenience: rotate turntable by degrees instead of raw steps
  if (cmd.startsWith("ROTATE:")) {
    float degrees = cmd.substring(7).toFloat();
    int steps = (int)(degrees * STEPS_PER_REV / 360.0);
    stepTurntable(steps);
    Serial.println("OK:ROTATE");
    return;
  }

  // ── PHOTO ─────────────────────────────────────────────────────
  if (cmd == "PHOTO") {
    // Was a GPIO pulse on the now-removed CAM_TRIGGER pin; migrated to a
    // Serial1 text trigger now that encoders are out of this build and
    // pins 18/19 are free. External command shape unchanged (still
    // PHOTO -> OK:PHOTO), so ArduinoSerial/CameraInterface on the Python
    // side needed no change. Same non-relay pattern as CAM_ARM_TRIGGER --
    // this sends the trigger only, the actual JPEG is still fetched over
    // WiFi HTTP by whatever's consuming it.
    Serial1.print("CAPTURE\n");
    Serial.println("OK:PHOTO");
    return;
  }

  // ── CAM_ARM_TRIGGER ──────────────────────────────────────────
  // Sends a short text trigger to the arm-angle ESP32-CAM over its real
  // hardware UART (Serial2). Does NOT relay image bytes back -- the
  // ESP32-CAM's own firmware needs to serve the resulting frame over its
  // own WiFi HTTP endpoint, same pattern as the undercarriage camera. See
  // integration note #6 for why this doesn't try to move JPEG data over
  // the UART link itself.
  if (cmd == "CAM_ARM_TRIGGER") {
    Serial2.print("CAPTURE\n");
    Serial.println("OK:CAM_ARM_TRIGGER");
    return;
  }

  // ── GPS_STATUS ───────────────────────────────────────────────
  // Reports whatever TinyGPS++'s internal state currently holds -- parsed
  // continuously in the background (see loop()). Does not read Serial3
  // directly here, and Serial3 bytes are never echoed to Serial.
  if (cmd == "GPS_STATUS") {
    Serial.print("{\"fix\":");
    Serial.print(gps.location.isValid() ? "true" : "false");
    if (gps.location.isValid()) {
      Serial.print(",\"lat\":");  Serial.print(gps.location.lat(), 6);
      Serial.print(",\"lon\":");  Serial.print(gps.location.lng(), 6);
      Serial.print(",\"sats\":"); Serial.print(gps.satellites.value());
    }
    Serial.println("}");
    return;
  }

  // ── LED:QUAD:<N|E|S|W|OFF> ───────────────────────────────────
  if (cmd.startsWith("LED:QUAD:")) {
    String which = cmd.substring(9);
    allQuadLedsOff();
    if      (which == "N")   ledDriver.setPWM(LED_CH_N, 0, LED_PWM_FULL_ON);
    else if (which == "E")   ledDriver.setPWM(LED_CH_E, 0, LED_PWM_FULL_ON);
    else if (which == "S")   ledDriver.setPWM(LED_CH_S, 0, LED_PWM_FULL_ON);
    else if (which == "W")   ledDriver.setPWM(LED_CH_W, 0, LED_PWM_FULL_ON);
    else if (which == "OFF") { /* already all off above */ }
    else { Serial.println("ERR:LED_QUAD_SYNTAX"); return; }
    Serial.println("OK:LED:QUAD");
    return;
  }

  // ── LED:UV:ON / LED:UV:OFF ───────────────────────────────────
  if (cmd == "LED:UV:ON") {
    ledDriver.setPWM(LED_CH_UV, 0, LED_PWM_FULL_ON);
    Serial.println("OK:LED:UV");
    return;
  }
  if (cmd == "LED:UV:OFF") {
    ledDriver.setPWM(LED_CH_UV, 0, LED_PWM_OFF);
    Serial.println("OK:LED:UV");
    return;
  }

  // ── Unknown ───────────────────────────────────────────────────
  Serial.print("ERR:UNKNOWN_CMD:");
  Serial.println(cmd);
}

// ═══════════════════════════════════════════════════════════════
// MOTOR CONTROL (BTS7960)
// ═══════════════════════════════════════════════════════════════

void setMotors(int leftSpeed, int rightSpeed) {
  // BTS7960: drive RPWM for one direction, LPWM for the other; the
  // opposite pin must be held LOW (analogWrite(0) achieves this cleanly).
  leftSpeed  = constrain(leftSpeed,  -255, 255);
  rightSpeed = constrain(rightSpeed, -255, 255);

  if (leftSpeed >= 0) {
    analogWrite(L_RPWM, leftSpeed);
    analogWrite(L_LPWM, 0);
  } else {
    analogWrite(L_RPWM, 0);
    analogWrite(L_LPWM, -leftSpeed);
  }

  if (rightSpeed >= 0) {
    analogWrite(R_RPWM, rightSpeed);
    analogWrite(R_LPWM, 0);
  } else {
    analogWrite(R_RPWM, 0);
    analogWrite(R_LPWM, -rightSpeed);
  }
}

void stopMotors() {
  analogWrite(L_RPWM, 0); analogWrite(L_LPWM, 0);
  analogWrite(R_RPWM, 0); analogWrite(R_LPWM, 0);
}

// ═══════════════════════════════════════════════════════════════
// ARM CONTROL
// ═══════════════════════════════════════════════════════════════

void moveToPose(const int target[5], int speedFactor) {
  /*
   * Smoothly interpolate from current pose to target pose.
   * speedFactor: 1=fast, 2=normal, 3=slow
   *   Controls how many intermediate steps (50 x speedFactor)
   */
  int steps = 50 * speedFactor;

  for (int i = 0; i <= steps; i++) {
    float t = (float)i / steps;  // 0.0 to 1.0

    int a0 = currentPose[0] + (int)((target[0] - currentPose[0]) * t);
    int a1 = currentPose[1] + (int)((target[1] - currentPose[1]) * t);
    int a2 = currentPose[2] + (int)((target[2] - currentPose[2]) * t);
    int a3 = currentPose[3] + (int)((target[3] - currentPose[3]) * t);

    s0.write(constrain(a0, 0, 180));
    s1.write(constrain(a1, 0, 180));
    s2.write(constrain(a2, 0, 180));
    s3.write(constrain(a3, 0, 180));

    delay(20);  // 20ms per step
  }

  // Update current pose (all 5 including gripper)
  for (int i = 0; i < 4; i++) currentPose[i] = target[i];
  // Note: gripper (currentPose[4]) is controlled separately via GRIP command
}

void driveGripperTo(int angle) {
  // Slow move to protect fragile objects — same 15ms/degree pacing as
  // the original numeric protocol, just driven to a fixed calibrated
  // angle instead of an arbitrary caller-supplied one.
  angle = constrain(angle, 0, 180);
  int current = currentPose[4];
  int step    = (angle > current) ? 1 : -1;
  while (current != angle) {
    current += step;
    s4.write(current);
    delay(15);
  }
  currentPose[4] = angle;
}

// ═══════════════════════════════════════════════════════════════
// TURNTABLE STEPPER (NEMA17 / A4988, STEP/DIR)
// ═══════════════════════════════════════════════════════════════

void stepTurntable(int steps) {
  /*
   * Step the turntable by 'steps' positions.
   * Positive = clockwise, negative = counterclockwise.
   *
   * A4988 current-limit trimpot sets physical torque/heat behavior --
   * not controlled from firmware. Step pulse timing below (2ms) is a
   * starting point carried over from the previous stepper's timing;
   * NEMA17 can likely go faster, but verify against real load before
   * tightening it.
   */
  digitalWrite(TT_ENABLE, LOW);   // enable driver (active LOW)
  digitalWrite(TT_DIR, steps >= 0 ? HIGH : LOW);
  steps = abs(steps);

  for (int i = 0; i < steps; i++) {
    digitalWrite(TT_STEP, HIGH);
    delayMicroseconds(500);
    digitalWrite(TT_STEP, LOW);
    delay(2);
  }

  digitalWrite(TT_ENABLE, HIGH);  // disable driver -- no holding current
                                    // between moves (matches the previous
                                    // stepper's "power off coils" behavior)
}

// ═══════════════════════════════════════════════════════════════
// LED CONTROL (PCA9685)
// ═══════════════════════════════════════════════════════════════

void allQuadLedsOff() {
  ledDriver.setPWM(LED_CH_N, 0, LED_PWM_OFF);
  ledDriver.setPWM(LED_CH_E, 0, LED_PWM_OFF);
  ledDriver.setPWM(LED_CH_S, 0, LED_PWM_OFF);
  ledDriver.setPWM(LED_CH_W, 0, LED_PWM_OFF);
}

void allLedsOff() {
  allQuadLedsOff();
  ledDriver.setPWM(LED_CH_UV, 0, LED_PWM_OFF);
}

// ═══════════════════════════════════════════════════════════════
// SENSORS
// ═══════════════════════════════════════════════════════════════

float readUltrasonic(int trigPin, int echoPin) {
  /*
   * Read HC-SR04 ultrasonic sensor.
   * Returns distance in centimeters.
   * Returns -1 if no echo received (out of range).
   */
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  long duration = pulseIn(echoPin, HIGH, 30000);  // 30ms timeout
  if (duration == 0) return -1;

  return duration * 0.034 / 2.0;  // cm
}
