/**
 * main.cpp — ESP32-CAM AI Digital Microscope
 * Camera  : OV2640 (built-in on ESP32-CAM)
 * Stream  : MJPEG over HTTP via ESPAsyncWebServer
 * Upload  : Via ESP32 DevKit V1 as USB-UART bridge
 */

#include <Arduino.h>
#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <ArduinoJson.h>
#include "esp_camera.h"

// ── Wi-Fi Credentials ─────────────────────────────────────────
#define WIFI_SSID  "aswin kanna wifi"
#define WIFI_PASS  "aswin@kannaa"
#define AP_SSID    "MicroScope-AP"
#define AP_PASS    "scope1234"

// ── ESP32-CAM AI Thinker Pin Map ──────────────────────────────
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ── Globals ───────────────────────────────────────────────────
AsyncWebServer server(80);
static uint32_t g_frame_count   = 0;
static float    g_fps           = 0.0f;
static uint32_t g_fps_times[10] = {0};
static uint8_t  g_fps_idx       = 0;
static uint32_t g_start_ms      = 0;

// ── Shared JPEG buffer ────────────────────────────────────────
static uint8_t*  g_jpg_buf  = nullptr;
static size_t    g_jpg_len  = 0;
static SemaphoreHandle_t g_jpg_mutex = nullptr;

// ── FPS Counter ───────────────────────────────────────────────
static void update_fps() {
    uint32_t now = millis();
    g_fps_times[g_fps_idx % 10] = now;
    if (g_frame_count >= 10) {
        uint32_t oldest = g_fps_times[(g_fps_idx + 1) % 10];
        if (now > oldest) g_fps = 10000.0f / (float)(now - oldest);
    }
    g_fps_idx++;
    g_frame_count++;
}

// ── Capture into shared buffer ────────────────────────────────
static bool capture_to_buf() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) return false;

    if (xSemaphoreTake(g_jpg_mutex, pdMS_TO_TICKS(300)) == pdTRUE) {
        if (g_jpg_buf) free(g_jpg_buf);
        g_jpg_buf = (uint8_t*)malloc(fb->len);
        if (g_jpg_buf) {
            memcpy(g_jpg_buf, fb->buf, fb->len);
            g_jpg_len = fb->len;
        }
        xSemaphoreGive(g_jpg_mutex);
    }
    esp_camera_fb_return(fb);
    update_fps();
    return true;
}

// ── Camera Task — continuous capture ─────────────────────────
void task_capture(void* pv) {
    while (true) {
        capture_to_buf();
        vTaskDelay(pdMS_TO_TICKS(50));  // ~20 FPS cap
    }
}

// ── Camera Init ───────────────────────────────────────────────
bool camera_init() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;

    if (psramFound()) {
        config.frame_size   = FRAMESIZE_VGA;
        config.jpeg_quality = 12;
        config.fb_count     = 2;
        Serial.println("[CAM] PSRAM found — VGA mode");
    } else {
        config.frame_size   = FRAMESIZE_QVGA;
        config.jpeg_quality = 15;
        config.fb_count     = 1;
        Serial.println("[CAM] No PSRAM — QVGA mode");
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("[CAM] Init failed: 0x%x\n", err);
        return false;
    }

    sensor_t* s = esp_camera_sensor_get();
    s->set_quality(s, 12);
    s->set_brightness(s, 0);
    s->set_contrast(s, 0);
    s->set_saturation(s, 0);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_exposure_ctrl(s, 1);
    s->set_gain_ctrl(s, 1);

    Serial.println("[CAM] OV2640 init OK");
    return true;
}

// ── Wi-Fi Init ────────────────────────────────────────────────
void wifi_init() {
    WiFi.mode(WIFI_AP_STA);
    WiFi.softAP(AP_SSID, AP_PASS);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    Serial.print("[WiFi] Connecting");

    uint8_t attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 10) {
        delay(1000); Serial.print("."); attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[WiFi] Connected — IP: %s\n",
                      WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\n[WiFi] STA failed — AP fallback");
        WiFi.mode(WIFI_AP);
        WiFi.softAP(AP_SSID, AP_PASS);
        Serial.printf("[WiFi] AP IP: %s\n",
                      WiFi.softAPIP().toString().c_str());
    }
}

// ── Routes ────────────────────────────────────────────────────
void setup_routes() {

    // GET /stream — serves latest JPEG from shared buffer
    server.on("/stream", HTTP_GET, [](AsyncWebServerRequest* req) {
        if (xSemaphoreTake(g_jpg_mutex, pdMS_TO_TICKS(300)) == pdTRUE) {
            if (!g_jpg_buf || g_jpg_len == 0) {
                xSemaphoreGive(g_jpg_mutex);
                req->send(503, "text/plain", "No frame yet");
                return;
            }
            // Copy buffer so we can release mutex quickly
            size_t len   = g_jpg_len;
            uint8_t* buf = (uint8_t*)malloc(len);
            if (buf) memcpy(buf, g_jpg_buf, len);
            xSemaphoreGive(g_jpg_mutex);

            if (!buf) {
                req->send(503, "text/plain", "Memory error");
                return;
            }

            AsyncWebServerResponse* resp =
                req->beginResponse_P(200, "image/jpeg", buf, len);
            resp->addHeader("Access-Control-Allow-Origin", "*");
            resp->addHeader("Cache-Control", "no-cache, no-store");
            req->send(resp);
            free(buf);
        } else {
            req->send(503, "text/plain", "Busy");
        }
    });

    // GET /capture — same as stream but with download header
    server.on("/capture", HTTP_GET, [](AsyncWebServerRequest* req) {
        if (xSemaphoreTake(g_jpg_mutex, pdMS_TO_TICKS(300)) == pdTRUE) {
            if (!g_jpg_buf || g_jpg_len == 0) {
                xSemaphoreGive(g_jpg_mutex);
                req->send(503, "text/plain", "No frame yet");
                return;
            }
            size_t len   = g_jpg_len;
            uint8_t* buf = (uint8_t*)malloc(len);
            if (buf) memcpy(buf, g_jpg_buf, len);
            xSemaphoreGive(g_jpg_mutex);

            if (!buf) {
                req->send(503, "text/plain", "Memory error");
                return;
            }

            AsyncWebServerResponse* resp =
                req->beginResponse_P(200, "image/jpeg", buf, len);
            resp->addHeader("Content-Disposition",
                            "attachment; filename=capture.jpg");
            resp->addHeader("Access-Control-Allow-Origin", "*");
            req->send(resp);
            free(buf);
            Serial.println("[CAM] Capture served");
        } else {
            req->send(503, "text/plain", "Busy");
        }
    });

    // GET /status — JSON
    server.on("/status", HTTP_GET, [](AsyncWebServerRequest* req) {
        StaticJsonDocument<256> doc;
        doc["wifi"]        = WiFi.status() == WL_CONNECTED ? "STA" : "AP";
        doc["ip"]          = WiFi.status() == WL_CONNECTED ?
                             WiFi.localIP().toString() :
                             WiFi.softAPIP().toString();
        doc["fps"]         = g_fps;
        doc["frame_count"] = g_frame_count;
        doc["heap_free"]   = ESP.getFreeHeap();
        doc["psram"]       = psramFound();
        doc["uptime_s"]    = (millis() - g_start_ms) / 1000;
        String body;
        serializeJson(doc, body);
        req->send(200, "application/json", body);
    });

    // POST /settings
    server.on("/settings", HTTP_POST,
        [](AsyncWebServerRequest* req) {},
        nullptr,
        [](AsyncWebServerRequest* req, uint8_t* data,
           size_t len, size_t, size_t) {
            StaticJsonDocument<128> doc;
            if (deserializeJson(doc, data, len) == DeserializationError::Ok) {
                sensor_t* s = esp_camera_sensor_get();
                if (doc.containsKey("brightness"))
                    s->set_brightness(s, doc["brightness"].as<int>());
                if (doc.containsKey("contrast"))
                    s->set_contrast(s, doc["contrast"].as<int>());
                if (doc.containsKey("quality"))
                    s->set_quality(s, doc["quality"].as<int>());
            }
            req->send(200, "application/json", "{\"status\":\"ok\"}");
        }
    );

    server.onNotFound([](AsyncWebServerRequest* req) {
        req->send(404, "text/plain", "Not found");
    });
}

// ── Setup ─────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("\n=== ESP32-CAM AI Digital Microscope ===");
    g_start_ms = millis();

    pinMode(4, OUTPUT);
    digitalWrite(4, LOW);

    Serial.printf("[SYS] Free heap  : %d bytes\n", ESP.getFreeHeap());
    Serial.printf("[SYS] PSRAM found: %s\n", psramFound() ? "YES" : "NO");

    g_jpg_mutex = xSemaphoreCreateMutex();

    if (!camera_init()) {
        Serial.println("[FATAL] Camera init failed");
        while (true) {
            digitalWrite(4, HIGH); delay(200);
            digitalWrite(4, LOW);  delay(200);
        }
    }

    // Warm up camera
    for (int i = 0; i < 3; i++) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) esp_camera_fb_return(fb);
        delay(100);
    }
    Serial.println("[CAM] Warm up done");

    wifi_init();
    setup_routes();
    server.begin();

    // Start capture task on Core 0
    xTaskCreatePinnedToCore(
        task_capture, "CaptureTask",
        4096, nullptr, 1, nullptr, 0
    );

    Serial.println("[HTTP] Server started on port 80");
    Serial.println("[MAIN] System running");
}

// ── Loop ──────────────────────────────────────────────────────
void loop() {
    static uint32_t last_print = 0;
    if (millis() - last_print > 5000) {
        Serial.printf(
            "[STATUS] FPS: %.1f | Frames: %lu | Heap: %lu | Uptime: %lus\n",
            g_fps, g_frame_count,
            ESP.getFreeHeap(),
            (millis() - g_start_ms) / 1000
        );
        last_print = millis();
    }
    delay(100);
}