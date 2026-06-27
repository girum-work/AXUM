/*
 * AXUM ROVER — ESP32-CAM Firmware
 * =================================
 * Dual-mode camera:
 *   Mode 1 — WiFi stream: streams MJPEG over HTTP for live dashboard view
 *   Mode 2 — Trigger capture: saves JPEG to SD card on trigger pin pulse
 *             OR serves single JPEG via HTTP /capture endpoint
 *
 * WiFi stream URL:  http://<IP>/stream    (for dashboard live feed)
 * Single capture:   http://<IP>/capture   (for photogrammetry photos)
 * Status check:     http://<IP>/status
 *
 * Hardware connections:
 *   GPIO 13 → Arduino pin 30 (trigger input — HIGH pulse = capture)
 *   GPIO 4  → LED flash (built-in on AI Thinker board)
 *
 * Setup:
 *   1. Set WIFI_SSID and WIFI_PASSWORD below
 *   2. Upload to ESP32-CAM
 *   3. Open Serial Monitor at 115200 baud
 *   4. Note the IP address printed — put it in config.py as ESP32_CAM_IP
 */

#include "esp_camera.h"
#include "esp_http_server.h"
#include <WiFi.h>
#include "SD_MMC.h"
#include "driver/rtc_io.h"

// ── WiFi credentials ───────────────────────────────────────────
// CHANGE THESE to your actual WiFi network
#define WIFI_SSID     "YOUR_WIFI_SSID"
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"

// ── Pin definitions (AI Thinker ESP32-CAM board) ───────────────
#define CAM_PIN_PWDN    32
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK     0
#define CAM_PIN_SIOD    26
#define CAM_PIN_SIOC    27
#define CAM_PIN_D7      35
#define CAM_PIN_D6      34
#define CAM_PIN_D5      39
#define CAM_PIN_D4      38
#define CAM_PIN_D3      37
#define CAM_PIN_D2      36
#define CAM_PIN_D1       5
#define CAM_PIN_D0       4
#define CAM_PIN_VSYNC   25
#define CAM_PIN_HREF    23
#define CAM_PIN_PCLK    22

#define TRIGGER_PIN     13   // receives HIGH pulse from Arduino pin 30
#define FLASH_LED_PIN    4   // built-in LED flash

// ── State ──────────────────────────────────────────────────────
static bool     captureRequested = false;
static int      photoIndex       = 0;
static httpd_handle_t streamServer = NULL;
static httpd_handle_t captureServer = NULL;

// ═══════════════════════════════════════════════════════════════
// CAMERA INITIALIZATION
// ═══════════════════════════════════════════════════════════════

bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = CAM_PIN_D0;
  config.pin_d1       = CAM_PIN_D1;
  config.pin_d2       = CAM_PIN_D2;
  config.pin_d3       = CAM_PIN_D3;
  config.pin_d4       = CAM_PIN_D4;
  config.pin_d5       = CAM_PIN_D5;
  config.pin_d6       = CAM_PIN_D6;
  config.pin_d7       = CAM_PIN_D7;
  config.pin_xclk     = CAM_PIN_XCLK;
  config.pin_pclk     = CAM_PIN_PCLK;
  config.pin_vsync    = CAM_PIN_VSYNC;
  config.pin_href     = CAM_PIN_HREF;
  config.pin_sccb_sda = CAM_PIN_SIOD;
  config.pin_sccb_scl = CAM_PIN_SIOC;
  config.pin_pwdn     = CAM_PIN_PWDN;
  config.pin_reset    = CAM_PIN_RESET;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Higher resolution for photogrammetry captures
  config.frame_size   = FRAMESIZE_UXGA;   // 1600×1200
  config.jpeg_quality = 12;               // 0-63, lower=better
  config.fb_count     = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }

  // Tune sensor settings
  sensor_t* s = esp_camera_sensor_get();
  s->set_brightness(s, 0);
  s->set_contrast(s, 0);
  s->set_saturation(s, 0);
  s->set_sharpness(s, 0);
  s->set_whitebal(s, 1);         // auto white balance
  s->set_awb_gain(s, 1);
  s->set_exposure_ctrl(s, 1);    // auto exposure
  s->set_aec2(s, 1);
  s->set_ae_level(s, 0);
  s->set_gain_ctrl(s, 1);        // auto gain
  s->set_agc_gain(s, 0);
  s->set_gainceiling(s, (gainceiling_t)2);
  s->set_bpc(s, 0);              // black pixel correction
  s->set_wpc(s, 1);              // white pixel correction
  s->set_raw_gma(s, 1);
  s->set_lenc(s, 1);             // lens correction
  s->set_hmirror(s, 0);
  s->set_vflip(s, 0);
  s->set_dcw(s, 1);
  s->set_colorbar(s, 0);

  return true;
}

// ═══════════════════════════════════════════════════════════════
// HTTP HANDLERS
// ═══════════════════════════════════════════════════════════════

// MJPEG stream boundary
#define STREAM_CONTENT_TYPE "multipart/x-mixed-replace;boundary=frame"
#define STREAM_BOUNDARY     "\r\n--frame\r\n"
#define STREAM_PART         "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n"

// Stream handler: serves continuous MJPEG stream
static esp_err_t streamHandler(httpd_req_t* req) {
  camera_fb_t* fb   = NULL;
  esp_err_t    res  = ESP_OK;
  char         part_buf[64];

  res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;

  // Stream indefinitely until client disconnects
  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Frame capture failed");
      res = ESP_FAIL;
      break;
    }

    // Send boundary
    res = httpd_resp_send_chunk(req, STREAM_BOUNDARY,
                                strlen(STREAM_BOUNDARY));
    if (res != ESP_OK) break;

    // Send part header with content length
    size_t hlen = snprintf(part_buf, sizeof(part_buf),
                           STREAM_PART, fb->len);
    res = httpd_resp_send_chunk(req, part_buf, hlen);
    if (res != ESP_OK) break;

    // Send JPEG data
    res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    if (res != ESP_OK) break;
  }

  return res;
}

// Capture handler: serves single JPEG on /capture
static esp_err_t captureHandler(httpd_req_t* req) {
  // Flash briefly for better lighting
  digitalWrite(FLASH_LED_PIN, HIGH);
  delay(50);

  camera_fb_t* fb = esp_camera_fb_get();
  digitalWrite(FLASH_LED_PIN, LOW);

  if (!fb) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }

  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Content-Disposition",
                     "inline; filename=capture.jpg");
  httpd_resp_send(req, (const char*)fb->buf, fb->len);
  esp_camera_fb_return(fb);

  photoIndex++;
  Serial.printf("Capture served: photo #%d\n", photoIndex);
  return ESP_OK;
}

// Status handler: returns JSON with IP and photo count
static esp_err_t statusHandler(httpd_req_t* req) {
  char json[128];
  snprintf(json, sizeof(json),
           "{\"ip\":\"%s\",\"photos\":%d,\"heap\":%lu}",
           WiFi.localIP().toString().c_str(),
           photoIndex,
           (unsigned long)ESP.getFreeHeap());
  httpd_resp_set_type(req, "application/json");
  httpd_resp_send(req, json, strlen(json));
  return ESP_OK;
}

// ═══════════════════════════════════════════════════════════════
// SERVER STARTUP
// ═══════════════════════════════════════════════════════════════

void startServers() {
  // Stream server on port 81
  httpd_config_t streamConfig = HTTPD_DEFAULT_CONFIG();
  streamConfig.server_port    = 81;
  streamConfig.ctrl_port      = 32769;

  httpd_uri_t streamUri = {
    .uri      = "/stream",
    .method   = HTTP_GET,
    .handler  = streamHandler,
    .user_ctx = NULL
  };

  if (httpd_start(&streamServer, &streamConfig) == ESP_OK) {
    httpd_register_uri_handler(streamServer, &streamUri);
    Serial.println("Stream server started on port 81");
  }

  // Capture server on port 80
  httpd_config_t captureConfig = HTTPD_DEFAULT_CONFIG();
  captureConfig.server_port    = 80;

  httpd_uri_t captureUri = {
    .uri      = "/capture",
    .method   = HTTP_GET,
    .handler  = captureHandler,
    .user_ctx = NULL
  };
  httpd_uri_t statusUri = {
    .uri      = "/status",
    .method   = HTTP_GET,
    .handler  = statusHandler,
    .user_ctx = NULL
  };

  if (httpd_start(&captureServer, &captureConfig) == ESP_OK) {
    httpd_register_uri_handler(captureServer, &captureUri);
    httpd_register_uri_handler(captureServer, &statusUri);
    Serial.println("Capture server started on port 80");
  }
}

// ═══════════════════════════════════════════════════════════════
// TRIGGER PIN HANDLER
// ═══════════════════════════════════════════════════════════════

void IRAM_ATTR onTriggerPulse() {
  /*
   * Interrupt: called when Arduino sends a HIGH pulse on trigger pin.
   * Sets a flag — actual capture happens in loop() to keep ISR short.
   */
  captureRequested = true;
}

void handleTriggerCapture() {
  /*
   * Called from loop() when captureRequested is true.
   * Saves JPEG to SD card with sequential filename.
   */
  captureRequested = false;

  // Flash briefly
  digitalWrite(FLASH_LED_PIN, HIGH);
  delay(30);

  camera_fb_t* fb = esp_camera_fb_get();
  digitalWrite(FLASH_LED_PIN, LOW);

  if (!fb) {
    Serial.println("Trigger capture failed: no frame");
    return;
  }

  // Save to SD card
  char filename[32];
  snprintf(filename, sizeof(filename), "/photo_%04d.jpg", photoIndex);

  File file = SD_MMC.open(filename, FILE_WRITE);
  if (file) {
    file.write(fb->buf, fb->len);
    file.close();
    Serial.printf("Saved: %s (%u bytes)\n", filename, fb->len);
    photoIndex++;
  } else {
    Serial.printf("SD write failed: %s\n", filename);
  }

  esp_camera_fb_return(fb);
}

// ═══════════════════════════════════════════════════════════════
// SETUP & LOOP
// ═══════════════════════════════════════════════════════════════

void setup() {
  Serial.begin(115200);
  Serial.println("\nAXUM ROVER — ESP32-CAM starting...");

  // Flash LED pin
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  // Trigger pin (input from Arduino)
  pinMode(TRIGGER_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(TRIGGER_PIN),
                  onTriggerPulse, RISING);

  // Initialize camera
  if (!initCamera()) {
    Serial.println("FATAL: Camera init failed");
    while (true) {
      digitalWrite(FLASH_LED_PIN, !digitalRead(FLASH_LED_PIN));
      delay(200);  // fast blink = error
    }
  }
  Serial.println("Camera OK");

  // Initialize SD card
  if (!SD_MMC.begin()) {
    Serial.println("WARNING: SD card not found — trigger capture disabled");
    Serial.println("HTTP capture (/capture) still works via WiFi");
  } else {
    Serial.println("SD card OK");
  }

  // Connect to WiFi
  Serial.printf("Connecting to WiFi: %s\n", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  WiFi.setSleep(false);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\nWiFi connection failed!");
    Serial.println("Check SSID and password in firmware");
    // Continue without WiFi — trigger capture to SD still works
  } else {
    Serial.printf("\nConnected! IP: %s\n", WiFi.localIP().toString().c_str());
    Serial.printf("Stream:  http://%s:81/stream\n",
                  WiFi.localIP().toString().c_str());
    Serial.printf("Capture: http://%s/capture\n",
                  WiFi.localIP().toString().c_str());
    Serial.printf("Status:  http://%s/status\n",
                  WiFi.localIP().toString().c_str());

    // Start HTTP servers
    startServers();
  }

  // Signal ready: 3 slow flashes
  for (int i = 0; i < 3; i++) {
    digitalWrite(FLASH_LED_PIN, HIGH); delay(200);
    digitalWrite(FLASH_LED_PIN, LOW);  delay(200);
  }

  Serial.println("ESP32-CAM ready");
}

void loop() {
  // Handle trigger capture from Arduino
  if (captureRequested) {
    handleTriggerCapture();
  }
  delay(10);
}