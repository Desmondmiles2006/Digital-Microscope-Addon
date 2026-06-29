# ESP32-OV7670 AI Digital Microscope Add-on

A wireless, AI-enabled digital microscope add-on built with an ESP32 Dev Module
and OV7670 camera. Streams live MJPEG video over Wi-Fi to a PyQt5 desktop app
with Claude Vision AI analysis and a Model Context Protocol (MCP) server.

---

## Wiring Diagram
```
ESP32 Dev Module          OV7670 Camera
─────────────────         ──────────────
3.3V  ─────────────────── 3.3V
GND   ─────────────────── GND
GPIO 0 (LEDC PWM) ─────── XCLK
GPIO 4  ────────────────── D0
GPIO 36 (input only) ───── D1
GPIO 2  ────────────────── D2
GPIO 15 ────────────────── D3
GPIO 12 ────────────────── D4
GPIO 13 ────────────────── D5
GPIO 14 ────────────────── D6
GPIO 27 ────────────────── D7
GPIO 25 ────────────────── PCLK
GPIO 26 ────────────────── HREF
GPIO 34 (input only) ───── VSYNC
GPIO 21 (SDA) ──────────── SIOD
GPIO 22 (SCL) ──────────── SIOC
GPIO 32 ────────────────── PWDN  (pull LOW to run)
GPIO 33 ────────────────── RESET (pull HIGH to run)

OLED SSD1306 (I2C)
GPIO 21 (SDA) ──── SDA
GPIO 22 (SCL) ──── SCL
3.3V  ───────────── VCC
GND   ───────────── GND
```

> ⚠️ OV7670 is 3.3V only. Never connect to 5V.

---

## Firmware Setup (PlatformIO)

1. Install VS Code + PlatformIO extension.
2. Clone this repo and open the root folder.
3. Edit `firmware/src/streamer.h` — set your `WIFI_SSID` and `WIFI_PASS`.
4. Build and flash:
```bash
   pio run --target upload
   pio device monitor --baud 115200
```
5. Note the IP address printed in Serial Monitor.

---

## PC App Setup
```bash
pip install -r requirements.txt
python pc_app/main.py
```

In the Settings tab, enter the ESP32 IP address and click Connect.

---

## MCP Server Setup
```bash
pip install anthropic requests
```

Edit `mcp_server/config.json` — set `esp32_ip` and `anthropic_api_key`.

Add to Claude Desktop `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "microscope": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server/server.py"]
    }
  }
}
```

Restart Claude Desktop. The microscope tools will appear in the tools panel.

---

## Library Install Reference

| Library | Install |
|---------|---------|
| ESPAsyncWebServer | PlatformIO lib_deps (auto) |
| Adafruit SSD1306 | PlatformIO lib_deps (auto) |
| anthropic | `pip install anthropic` |
| ultralytics | `pip install ultralytics` |
| reportlab | `pip install reportlab` |
| PyQt5 | `pip install PyQt5` |

& "D:\micro_scope\venv_gpu\Scripts\Activate.ps1"
192.168.4.1