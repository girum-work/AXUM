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
 *   ATTITUDE                            {"ok":bool,"roll":...,"pitch":...,"yaw":...,"biased":bool,"hz":...}
 *   IMU:CALIBRATE                       OK:IMU:CALIBRATE
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
 *
 * 8. GY-80 9-axis IMU (I2C, confirmed hardware -- supersedes the MPU6050
 *    exclusion in note 7, which was never confirmed and is not fitted).
 *
 *    The GY-80 is three independent dies on one breakout, not a fused
 *    sensor like the BNO055 -- there is no on-chip DMP and no on-chip
 *    calibration status byte, so orientation has to be fused in firmware:
 *      - ADXL345   @ 0x53  accelerometer  (gravity reference: roll/pitch)
 *      - L3G4200D  @ 0x69  gyroscope      (rate integration: all axes)
 *      - HMC5883L  @ 0x1E  magnetometer   (heading reference: yaw)
 *      - BMP085    @ 0x77  barometer      (not read -- altitude unused)
 *
 *    Fusion is Mahony, not Madgwick. Both were considered; Mahony is the
 *    right pick for an ATmega2560 because its correction step is a plain
 *    cross-product with a PI controller, whereas Madgwick runs a gradient
 *    -descent step with an inverse square root every iteration. On an
 *    8-bit AVR with no FPU that difference is the difference between
 *    holding 100 Hz and not. Accuracy is equivalent for a ground vehicle
 *    that never leaves roughly-level attitudes.
 *
 *    Read directly over Wire rather than via a driver library: the three
 *    dies need ~8 register writes total to configure, and pulling in
 *    three Adafruit libraries to save those writes would cost more flash
 *    than the whole fusion filter.
 *
 *    Two gotchas worth writing down, both of which produce plausible-
 *    looking-but-wrong output rather than an obvious failure:
 *      - HMC5883L returns its axes in X, Z, Y order (not X, Y, Z) and
 *        big-endian, unlike the other two dies which are little-endian
 *        X, Y, Z. Reading it as XYZ silently swaps two axes and yaw
 *        drifts in a way that looks like a calibration problem.
 *      - The gyro must be biased at rest. An uncalibrated L3G4200D
 *        typically sits at tens of LSB of zero-rate offset, which the
 *        integrator turns into visible yaw creep within a minute.
 *        IMU:CALIBRATE re-measures the bias; it is also run once at boot,
 *        which assumes the rover is stationary at power-on.
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
// GY-80 IMU — I2C registers, Mahony AHRS state
// See integration note #8 for why this is hand-rolled and why Mahony.
// ═══════════════════════════════════════════════════════════════

const uint8_t ADXL345_ADDR   = 0x53;
const uint8_t ADXL345_POWER  = 0x2D;
const uint8_t ADXL345_FORMAT = 0x31;
const uint8_t ADXL345_BWRATE = 0x2C;
const uint8_t ADXL345_DATAX0 = 0x32;

const uint8_t L3G4200D_ADDR  = 0x69;
const uint8_t L3G4200D_CTRL1 = 0x20;
const uint8_t L3G4200D_CTRL4 = 0x23;
const uint8_t L3G4200D_OUTX  = 0x28;

const uint8_t HMC5883L_ADDR  = 0x1E;
const uint8_t HMC5883L_CFGA  = 0x00;
const uint8_t HMC5883L_CFGB  = 0x01;
const uint8_t HMC5883L_MODE  = 0x02;
const uint8_t HMC5883L_DATA  = 0x03;

// ADXL345 in full-resolution mode is 3.9 mg/LSB at every range.
const float ACCEL_LSB_G      = 0.0039f;
// L3G4200D at the +/-250 dps range is 8.75 mdps/LSB.
const float GYRO_LSB_DPS     = 0.00875f;
const float DEG_TO_RADIANS   = 0.017453293f;
const float RADIANS_TO_DEG   = 57.29577951f;

// Mahony PI gains. Kp sets how hard accel/mag pull the estimate back to
// the gravity/magnetic reference; Ki removes residual gyro bias. Ki is
// deliberately small -- a large Ki fights the explicit bias subtraction
// in imuCalibrate() and makes settling oscillate.
const float MAHONY_TWO_KP    = 2.0f * 0.9f;
const float MAHONY_TWO_KI    = 2.0f * 0.003f;

const unsigned long IMU_PERIOD_US = 10000UL;   // 100 Hz fusion rate

float q0 = 1.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
float integralFBx = 0.0f, integralFBy = 0.0f, integralFBz = 0.0f;
float gyroBiasX = 0.0f, gyroBiasY = 0.0f, gyroBiasZ = 0.0f;
float imuRoll = 0.0f, imuPitch = 0.0f, imuYaw = 0.0f;
float imuRateHz = 0.0f;
bool  imuPresent = false;
bool  imuBiased  = false;
unsigned long imuLastUpdateUs = 0;

// ═══════════════════════════════════════════════════════════════
// GY-80 IMU — low-level I2C
// ═══════════════════════════════════════════════════════════════

void i2cWrite(uint8_t addr, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

bool i2cPresent(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

// Reads `count` bytes into buf. Returns false on a short read so callers
// can skip the fusion step rather than integrating whatever was left in
// the buffer from the previous sample.
bool i2cReadBytes(uint8_t addr, uint8_t reg, uint8_t *buf, uint8_t count) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(addr, count) != count) return false;
  for (uint8_t i = 0; i < count; i++) buf[i] = Wire.read();
  return true;
}

bool readAccel(float *ax, float *ay, float *az) {
  uint8_t b[6];
  if (!i2cReadBytes(ADXL345_ADDR, ADXL345_DATAX0, b, 6)) return false;
  *ax = (int16_t)(b[0] | (b[1] << 8)) * ACCEL_LSB_G;
  *ay = (int16_t)(b[2] | (b[3] << 8)) * ACCEL_LSB_G;
  *az = (int16_t)(b[4] | (b[5] << 8)) * ACCEL_LSB_G;
  return true;
}

bool readGyroRaw(float *gx, float *gy, float *gz) {
  uint8_t b[6];
  // Bit 7 of the register address is L3G4200D's auto-increment flag;
  // without it every byte of the burst comes back from OUT_X_L.
  if (!i2cReadBytes(L3G4200D_ADDR, L3G4200D_OUTX | 0x80, b, 6)) return false;
  *gx = (int16_t)(b[0] | (b[1] << 8)) * GYRO_LSB_DPS;
  *gy = (int16_t)(b[2] | (b[3] << 8)) * GYRO_LSB_DPS;
  *gz = (int16_t)(b[4] | (b[5] << 8)) * GYRO_LSB_DPS;
  return true;
}

bool readMag(float *mx, float *my, float *mz) {
  uint8_t b[6];
  if (!i2cReadBytes(HMC5883L_ADDR, HMC5883L_DATA, b, 6)) return false;
  // Big-endian, and the axis order on the wire is X, Z, Y -- see note #8.
  *mx = (int16_t)((b[0] << 8) | b[1]);
  *mz = (int16_t)((b[2] << 8) | b[3]);
  *my = (int16_t)((b[4] << 8) | b[5]);
  return true;
}

// Averages the gyro at rest to find zero-rate offset. Blocking, ~200 ms.
// The rover must be stationary; there is no way to detect that it isn't,
// so a calibrate taken while moving will bake motion into the bias.
void imuCalibrate() {
  if (!imuPresent) return;
  const int samples = 200;
  float sx = 0, sy = 0, sz = 0;
  int taken = 0;
  for (int i = 0; i < samples; i++) {
    float gx, gy, gz;
    if (readGyroRaw(&gx, &gy, &gz)) {
      sx += gx; sy += gy; sz += gz;
      taken++;
    }
    delay(1);
  }
  if (taken == 0) return;
  gyroBiasX = sx / taken;
  gyroBiasY = sy / taken;
  gyroBiasZ = sz / taken;
  integralFBx = integralFBy = integralFBz = 0.0f;
  imuBiased = true;
}

void imuBegin() {
  imuPresent = i2cPresent(ADXL345_ADDR)
            && i2cPresent(L3G4200D_ADDR)
            && i2cPresent(HMC5883L_ADDR);
  if (!imuPresent) return;

  i2cWrite(ADXL345_ADDR, ADXL345_BWRATE, 0x0A);  // 100 Hz output
  i2cWrite(ADXL345_ADDR, ADXL345_FORMAT, 0x0B);  // full res, +/-16 g
  i2cWrite(ADXL345_ADDR, ADXL345_POWER,  0x08);  // leave standby

  i2cWrite(L3G4200D_ADDR, L3G4200D_CTRL1, 0x0F); // 100 Hz, all axes on
  i2cWrite(L3G4200D_ADDR, L3G4200D_CTRL4, 0x00); // +/-250 dps

  i2cWrite(HMC5883L_ADDR, HMC5883L_CFGA, 0x70);  // 8 avg, 15 Hz
  i2cWrite(HMC5883L_ADDR, HMC5883L_CFGB, 0xA0);  // +/-4.7 Ga
  i2cWrite(HMC5883L_ADDR, HMC5883L_MODE, 0x00);  // continuous

  delay(20);
  imuCalibrate();
  imuLastUpdateUs = micros();
}

// Mahony AHRS. Called from loop() at IMU_PERIOD_US; skips silently when
// the IMU is absent so a rover built without one still runs everything
// else. Degrades to 6-axis (accel + gyro) if the magnetometer reads zero,
// which is what happens near a motor that has magnetised the compass.
void imuUpdate() {
  if (!imuPresent) return;

  unsigned long now = micros();
  unsigned long elapsed = now - imuLastUpdateUs;
  if (elapsed < IMU_PERIOD_US) return;
  imuLastUpdateUs = now;

  float dt = elapsed * 1e-6f;
  imuRateHz = 1.0f / dt;

  float ax, ay, az, gxDps, gyDps, gzDps, mx, my, mz;
  if (!readAccel(&ax, &ay, &az)) return;
  if (!readGyroRaw(&gxDps, &gyDps, &gzDps)) return;
  bool haveMag = readMag(&mx, &my, &mz);

  float gx = (gxDps - gyroBiasX) * DEG_TO_RADIANS;
  float gy = (gyDps - gyroBiasY) * DEG_TO_RADIANS;
  float gz = (gzDps - gyroBiasZ) * DEG_TO_RADIANS;

  float aNorm = sqrt(ax * ax + ay * ay + az * az);
  if (aNorm > 0.0f) {
    ax /= aNorm; ay /= aNorm; az /= aNorm;

    float mNorm = haveMag ? sqrt(mx * mx + my * my + mz * mz) : 0.0f;
    if (mNorm > 0.0f) { mx /= mNorm; my /= mNorm; mz /= mNorm; }
    else              { haveMag = false; }

    float q0q0 = q0 * q0, q0q1 = q0 * q1, q0q2 = q0 * q2, q0q3 = q0 * q3;
    float q1q1 = q1 * q1, q1q2 = q1 * q2, q1q3 = q1 * q3;
    float q2q2 = q2 * q2, q2q3 = q2 * q3, q3q3 = q3 * q3;

    float halfex, halfey, halfez;

    // Estimated gravity direction in the body frame.
    float halfvx = q1q3 - q0q2;
    float halfvy = q0q1 + q2q3;
    float halfvz = q0q0 - 0.5f + q3q3;

    if (haveMag) {
      // Rotate the measured field into earth frame, flatten it onto the
      // horizontal plane, then rotate the reference back into the body
      // frame -- this is what stops magnetic dip from leaking into
      // roll/pitch the way a naive 3-axis compare would.
      float hx = 2.0f * (mx * (0.5f - q2q2 - q3q3) + my * (q1q2 - q0q3)   + mz * (q1q3 + q0q2));
      float hy = 2.0f * (mx * (q1q2 + q0q3)        + my * (0.5f - q1q1 - q3q3) + mz * (q2q3 - q0q1));
      float bx = sqrt(hx * hx + hy * hy);
      float bz = 2.0f * (mx * (q1q3 - q0q2)        + my * (q2q3 + q0q1)   + mz * (0.5f - q1q1 - q2q2));

      float halfwx = bx * (0.5f - q2q2 - q3q3) + bz * (q1q3 - q0q2);
      float halfwy = bx * (q1q2 - q0q3)        + bz * (q0q1 + q2q3);
      float halfwz = bx * (q0q2 + q1q3)        + bz * (0.5f - q1q1 - q2q2);

      halfex = (ay * halfvz - az * halfvy) + (my * halfwz - mz * halfwy);
      halfey = (az * halfvx - ax * halfvz) + (mz * halfwx - mx * halfwz);
      halfez = (ax * halfvy - ay * halfvx) + (mx * halfwy - my * halfwx);
    } else {
      halfex = ay * halfvz - az * halfvy;
      halfey = az * halfvx - ax * halfvz;
      halfez = ax * halfvy - ay * halfvx;
    }

    if (MAHONY_TWO_KI > 0.0f) {
      integralFBx += MAHONY_TWO_KI * halfex * dt;
      integralFBy += MAHONY_TWO_KI * halfey * dt;
      integralFBz += MAHONY_TWO_KI * halfez * dt;
      gx += integralFBx; gy += integralFBy; gz += integralFBz;
    }

    gx += MAHONY_TWO_KP * halfex;
    gy += MAHONY_TWO_KP * halfey;
    gz += MAHONY_TWO_KP * halfez;
  }

  gx *= 0.5f * dt; gy *= 0.5f * dt; gz *= 0.5f * dt;
  float qa = q0, qb = q1, qc = q2;
  q0 += (-qb * gx - qc * gy - q3 * gz);
  q1 += ( qa * gx + qc * gz - q3 * gy);
  q2 += ( qa * gy - qb * gz + q3 * gx);
  q3 += ( qa * gz + qb * gy - qc * gx);

  float qNorm = sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
  if (qNorm <= 0.0f) return;
  q0 /= qNorm; q1 /= qNorm; q2 /= qNorm; q3 /= qNorm;

  imuRoll  = atan2(2.0f * (q0 * q1 + q2 * q3), 1.0f - 2.0f * (q1 * q1 + q2 * q2)) * RADIANS_TO_DEG;
  float sinPitch = 2.0f * (q0 * q2 - q3 * q1);
  sinPitch = constrain(sinPitch, -1.0f, 1.0f);
  imuPitch = asin(sinPitch) * RADIANS_TO_DEG;
  imuYaw   = atan2(2.0f * (q0 * q3 + q1 * q2), 1.0f - 2.0f * (q2 * q2 + q3 * q3)) * RADIANS_TO_DEG;
  if (imuYaw < 0.0f) imuYaw += 360.0f;   // report 0..360, not -180..180
}

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

  // GY-80 IMU -- shares the PCA9685's I2C bus, so it has to come after
  // Wire.begin(). Boot-time calibrate assumes the rover is stationary.
  imuBegin();

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

  // Orientation has to be integrated continuously, not sampled when
  // ATTITUDE is asked for -- a gyro only measures rate, so any gap in
  // the integration is rotation the estimate never sees.
  imuUpdate();

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

  // ── ATTITUDE ─────────────────────────────────────────────────
  // Data query, so JSON with no OK: prefix -- same shape as STATUS and
  // GPS_STATUS. Reports the running fusion estimate; it does not sample
  // the sensors here (see loop()).
  if (cmd == "ATTITUDE") {
    Serial.print("{\"ok\":");     Serial.print(imuPresent ? "true" : "false");
    Serial.print(",\"roll\":");   Serial.print(imuRoll, 2);
    Serial.print(",\"pitch\":");  Serial.print(imuPitch, 2);
    Serial.print(",\"yaw\":");    Serial.print(imuYaw, 2);
    Serial.print(",\"biased\":"); Serial.print(imuBiased ? "true" : "false");
    Serial.print(",\"hz\":");     Serial.print(imuRateHz, 1);
    Serial.println("}");
    return;
  }

  // ── IMU:CALIBRATE ────────────────────────────────────────────
  // Blocks ~200 ms while it averages the gyro at rest.
  if (cmd == "IMU:CALIBRATE") {
    if (!imuPresent) { Serial.println("ERR:IMU_ABSENT"); return; }
    imuCalibrate();
    Serial.println("OK:IMU:CALIBRATE");
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
