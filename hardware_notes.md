# OV7670 Hardware Notes & Tuning Guide

## Known Quirks

- **3.3V only** — Never connect OV7670 to 5V; it will be damaged immediately.
- **PWDN must be LOW to operate** — Pull GPIO 32 LOW in init before sending XCLK.
- **XCLK before RESET** — Start the LEDC clock before releasing RESET or the sensor
  will not initialize correctly.
- **GPIO 36 (D1) and GPIO 34 (VSYNC)** — Input-only pins on ESP32 WROOM-32. Do not
  configure as OUTPUT. These must remain INPUT-only.
- **No FIFO variant** — Without FIFO, all pixel data must be read in real time
  synchronized to PCLK. At 16 MHz XCLK with QVGA, you have ~125 ns per pixel —
  GPIO polling in Arduino loop() is too slow for full VGA; use QVGA (320x240).
- **I2C address** — OV7670 uses SCCB address 0x21 (write) / 0x42 (8-bit write address).
  Do not confuse with standard I2C 7-bit addressing.
- **COM7 reset (0x80)** — After sending a full register reset, wait at least 30 ms
  before writing further registers or values will not stick.

## Register Tuning Tips

| Goal | Register | Value |
|------|----------|-------|
| Flip image horizontally | MVFP (0x1E) | Set bit 5 |
| Flip vertically | MVFP (0x1E) | Set bit 4 |
| Disable night mode | COM11 (0x3B) | 0x00 |
| Enable night mode | COM11 (0x3B) | 0x80 |
| Set manual gain | COM8 (0x13) | Clear bit 2; then set GAIN (0x00) |
| Reduce frame rate | CLKRC (0x11) | Increase prescaler value |
| Force 15 FPS | CLKRC (0x11) | 0x03 (÷4) |

## White Balance Adjustment

The OV7670 supports automatic white balance (AWB) controlled by COM8 bit 1.

For manual white balance:
1. Disable AWB: `ov7670_write_reg(0x13, ov7670_read_reg(0x13) & ~0x02)`
2. Set blue gain:  `ov7670_write_reg(0x01, blue_val)`   // 0x00–0xFF
3. Set red gain:   `ov7670_write_reg(0x02, red_val)`    // 0x00–0xFF

Typical outdoor white: blue=0x40, red=0x60  
Typical indoor incandescent: blue=0x58, red=0x48

## Frame Rate Optimization

- **QVGA at ~15 FPS** is achievable with GPIO polling in ISR on ESP32 at 240 MHz.
- To increase FPS: reduce CLKRC prescaler, reduce image resolution.
- Pinning the camera task to Core 0 and disabling WiFi on Core 0 prevents
  interrupt latency spikes from WiFi causing pixel sampling errors.
- Use `CONFIG_FREERTOS_UNICORE=n` (default) to keep dual-core mode active.
- For higher throughput, consider I2S parallel input mode (requires custom driver
  not included in this Arduino build, but available in ESP-IDF).

## Oscilloscope Debug Points

- VSYNC: should pulse once per frame (~66 ms at 15 FPS)
- HREF: should go HIGH once per row, 240 times per frame
- PCLK: should be ~8 MHz (XCLK÷2 after internal PLL)