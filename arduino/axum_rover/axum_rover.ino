/*
 * AXUM ROVER — Main Arduino Firmware
 * ====================================
 * Controls:
 *   - 4× drive motors via 2× L298N motor drivers
 *   - 4× arm servos (base, shoulder, elbow, wrist) + 1× gripper servo
 *   - 1× 28BYJ-48 stepper motor (turntable)
 *   - NeoPixel LED ring
 *   - HC-SR04 ultrasonic sensors (front + side)
 *   - 1× encoder per side (wheel odometry)
 *
 * Communication: USB Serial at 115200 baud
 * Protocol: newline-terminated ASCII commands
 *
 * Command reference:
 *   DRIVE:<left_speed>,<right_speed>   speed: -255 to 255
 *   ARM:<s0>,<s1>,<s2>,<s3>            angles: 0 to 180 degrees
 *   GRIP:<angle>                        0=open, 90=closed
 *   STEP:<steps>                        turntable steps (+ = CW)
 *   LED:<r>,<g>,<b>                     NeoPixel color 0-255
 *   PHOTO                               trigger ESP32-CAM capture
 *   PING                                returns PONG (connection check)
 *   STOP                                emergency stop all motion
 *   STATUS                              returns sensor readings JSON
 */

#include <Servo.h>
#include <Stepper.h>

// ═══════════════════════════════════════════════════════════════
// PIN DEFINITIONS
// ═══════════════════════════════════════════════════════════════

// Drive motors — L298N #1 (LEFT side)
const int L_ENA = 44;   // PWM speed
const int L_IN1 = 7;
const int L_IN2 = 8;

// Drive motors — L298N #2 (RIGHT side)
const int R_ENA = 45;   // PWM speed
const int R_IN1 = 12;
const int R_IN2 = 13;

// Arm servos
const int PIN_S0 = 3;    // Base rotation (MG996R)
const int PIN_S1 = 4;    // Shoulder       (MG996R)
const int PIN_S2 = 6;    // Elbow          (MG90S)
const int PIN_S3 = 9;    // Wrist tilt     (SG90)
const int PIN_S4 = 10;   // Gripper        (SG90)

// Turntable stepper (28BYJ-48)
const int STEP_IN1 = 22;
const int STEP_IN2 = 23;
const int STEP_IN3 = 24;
const int STEP_IN4 = 25;

// NeoPixel LED ring
const int NEO_PIN  = 11;
const int NEO_COUNT = 12;  // number of LEDs in your ring

// Ultrasonic sensors
const int FRONT_TRIG = 26;
const int FRONT_ECHO = 27;
const int SIDE_TRIG  = 28;
const int SIDE_ECHO  = 29;

// ESP32-CAM trigger (HIGH pulse triggers capture)
const int CAM_TRIGGER = 30;

// Encoders (interrupt pins on Mega: 2, 3, 18, 19, 20, 21)
const int ENC_LEFT  = 18;
const int ENC_RIGHT = 19;

// ═══════════════════════════════════════════════════════════════
// OBJECTS
// ═══════════════════════════════════════════════════════════════

Servo s0, s1, s2, s3, s4;

// 28BYJ-48: 2048 steps per revolution in half-step mode
// Using Arduino Stepper library (4096 steps for full revolution)
// We'll manually step for precision
const int STEPS_PER_REV = 2048;

// Current arm pose [s0, s1, s2, s3, s4]
int currentPose[5] = {90, 30, 150, 90, 160};

// Encoder counts
volatile long encLeft  = 0;
volatile long encRight = 0;

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

  // Attach servos
  s0.attach(PIN_S0);
  s1.attach(PIN_S1);
  s2.attach(PIN_S2);
  s3.attach(PIN_S3);
  s4.attach(PIN_S4);

  // Drive motor pins
  pinMode(L_ENA, OUTPUT); pinMode(L_IN1, OUTPUT); pinMode(L_IN2, OUTPUT);
  pinMode(R_ENA, OUTPUT); pinMode(R_IN1, OUTPUT); pinMode(R_IN2, OUTPUT);

  // Stepper pins
  pinMode(STEP_IN1, OUTPUT); pinMode(STEP_IN2, OUTPUT);
  pinMode(STEP_IN3, OUTPUT); pinMode(STEP_IN4, OUTPUT);

  // Sensor pins
  pinMode(FRONT_TRIG, OUTPUT); pinMode(FRONT_ECHO, INPUT);
  pinMode(SIDE_TRIG,  OUTPUT); pinMode(SIDE_ECHO,  INPUT);

  // Camera trigger
  pinMode(CAM_TRIGGER, OUTPUT);
  digitalWrite(CAM_TRIGGER, LOW);

  // Encoder interrupts
  attachInterrupt(digitalPinToInterrupt(ENC_LEFT),  countLeft,  RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_RIGHT), countRight, RISING);

  // Move to park pose on startup
  moveToPose(POSE_PARK, 3);

  // Stop motors
  stopMotors();

  Serial.println("AXUM_READY");
}

// ═══════════════════════════════════════════════════════════════
// ENCODER ISRs
// ═══════════════════════════════════════════════════════════════

void countLeft()  { encLeft++;  }
void countRight() { encRight++; }

// ═══════════════════════════════════════════════════════════════
// MAIN LOOP — Command Parser
// ═══════════════════════════════════════════════════════════════

void loop() {
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
    Serial.print(",\"enc_l\":");  Serial.print(encLeft);
    Serial.print(",\"enc_r\":");  Serial.print(encRight);
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

  // ── GRIP:<angle> ──────────────────────────────────────────────
  if (cmd.startsWith("GRIP:")) {
    int angle = cmd.substring(5).toInt();
    angle = constrain(angle, 0, 180);

    // Slow the gripper close to protect fragile objects
    int current = currentPose[4];
    int step    = (angle > current) ? 1 : -1;
    while (current != angle) {
      current += step;
      s4.write(current);
      delay(15);  // 15ms per degree = slow close
    }
    currentPose[4] = angle;
    Serial.println("OK:GRIP");
    return;
  }

  // ── POSE:<name> ───────────────────────────────────────────────
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
    // 28BYJ-48: 2048 steps = 360 degrees
    int steps = (int)(degrees * STEPS_PER_REV / 360.0);
    stepTurntable(steps);
    Serial.println("OK:ROTATE");
    return;
  }

  // ── PHOTO ─────────────────────────────────────────────────────
  if (cmd == "PHOTO") {
    // Send a 100ms HIGH pulse to ESP32-CAM trigger pin
    digitalWrite(CAM_TRIGGER, HIGH);
    delay(100);
    digitalWrite(CAM_TRIGGER, LOW);
    Serial.println("OK:PHOTO");
    return;
  }

  // ── LED:<r>,<g>,<b> ───────────────────────────────────────────
  if (cmd.startsWith("LED:")) {
    // Simple PWM output on NEO_PIN for single-color LED rings
    // For NeoPixel, replace with Adafruit NeoPixel library calls
    String params = cmd.substring(4);
    // For now, just acknowledge — implement NeoPixel in next iteration
    Serial.println("OK:LED");
    return;
  }

  // ── Unknown ───────────────────────────────────────────────────
  Serial.print("ERR:UNKNOWN_CMD:");
  Serial.println(cmd);
}

// ═══════════════════════════════════════════════════════════════
// MOTOR CONTROL
// ═══════════════════════════════════════════════════════════════

void setMotors(int leftSpeed, int rightSpeed) {
  // Left motors
  if (leftSpeed >= 0) {
    digitalWrite(L_IN1, HIGH);
    digitalWrite(L_IN2, LOW);
  } else {
    digitalWrite(L_IN1, LOW);
    digitalWrite(L_IN2, HIGH);
    leftSpeed = -leftSpeed;
  }
  analogWrite(L_ENA, constrain(leftSpeed, 0, 255));

  // Right motors
  if (rightSpeed >= 0) {
    digitalWrite(R_IN1, HIGH);
    digitalWrite(R_IN2, LOW);
  } else {
    digitalWrite(R_IN1, LOW);
    digitalWrite(R_IN2, HIGH);
    rightSpeed = -rightSpeed;
  }
  analogWrite(R_ENA, constrain(rightSpeed, 0, 255));
}

void stopMotors() {
  analogWrite(L_ENA, 0);
  analogWrite(R_ENA, 0);
  digitalWrite(L_IN1, LOW); digitalWrite(L_IN2, LOW);
  digitalWrite(R_IN1, LOW); digitalWrite(R_IN2, LOW);
}

// ═══════════════════════════════════════════════════════════════
// ARM CONTROL
// ═══════════════════════════════════════════════════════════════

void moveToPose(const int target[5], int speedFactor) {
  /*
   * Smoothly interpolate from current pose to target pose.
   * speedFactor: 1=fast, 2=normal, 3=slow
   *   Controls how many intermediate steps (50 × speedFactor)
   */
  int steps = 50 * speedFactor;

  for (int i = 0; i <= steps; i++) {
    float t = (float)i / steps;  // 0.0 to 1.0

    // Ease-in-out: smoother motion at start and end
    // t = t*t*(3 - 2*t);  // uncomment for smoother motion

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

// ═══════════════════════════════════════════════════════════════
// TURNTABLE STEPPER
// ═══════════════════════════════════════════════════════════════

// 28BYJ-48 half-step sequence (8 steps per electrical cycle)
const int STEP_SEQ[8][4] = {
  {1, 0, 0, 0},
  {1, 1, 0, 0},
  {0, 1, 0, 0},
  {0, 1, 1, 0},
  {0, 0, 1, 0},
  {0, 0, 1, 1},
  {0, 0, 0, 1},
  {1, 0, 0, 1}
};

int stepIndex = 0;

void stepTurntable(int steps) {
  /*
   * Step the turntable by 'steps' positions.
   * Positive = clockwise, negative = counterclockwise.
   * 
   * Speed: 2ms per step = ~1 revolution per 4 seconds
   * For smoother rotation, increase delay (3–5ms)
   */
  int direction = (steps > 0) ? 1 : -1;
  steps = abs(steps);

  for (int i = 0; i < steps; i++) {
    stepIndex = (stepIndex + direction + 8) % 8;

    digitalWrite(STEP_IN1, STEP_SEQ[stepIndex][0]);
    digitalWrite(STEP_IN2, STEP_SEQ[stepIndex][1]);
    digitalWrite(STEP_IN3, STEP_SEQ[stepIndex][2]);
    digitalWrite(STEP_IN4, STEP_SEQ[stepIndex][3]);

    delay(2);  // step delay — adjust for speed vs torque
  }

  // Power off stepper coils to prevent heating
  // (28BYJ-48 runs hot if coils stay energized)
  digitalWrite(STEP_IN1, LOW);
  digitalWrite(STEP_IN2, LOW);
  digitalWrite(STEP_IN3, LOW);
  digitalWrite(STEP_IN4, LOW);
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