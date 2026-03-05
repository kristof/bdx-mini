# ESP32 Virtual Servo

This ESP32 firmware acts as a virtual Feetech servo (ID 40) on the serial bus, controlling:
- 2x GC9D01 0.71" round TFT displays (eyes with 6 expression modes)
- 2x PWM servos (antennas)
- 1x 3.3V LED (projector)

## Hardware Requirements

- ESP32 D1 Mini (or compatible)
- 2x GC9D01 0.71" round TFT displays (160x160, 8-pin)
- 2x Standard PWM servos (for antennas)
- 1x LED with appropriate resistor (for projector)

## Pin Connections

| Function | GPIO |
|----------|------|
| Serial RX (servo bus) | 16 |
| Left Antenna PWM | 25 |
| Right Antenna PWM | 26 |
| Left TFT CS | 5 |
| Right TFT CS | 15 (TD0) |
| TFT DC | 2 |
| TFT CLK | 14 (TMS) |
| TFT MOSI | 4 |
| TFT RST | 13 (TCK) |
| TFT Backlight | 21 |
| Projector LED | 22 |

## TFT_eSPI Library Setup

1. Install **TFT_eSPI** library via Arduino Library Manager

2. Edit the library's `User_Setup.h` file (usually in `~/Arduino/libraries/TFT_eSPI/`):

```cpp
// Comment out the default driver and enable GC9A01 (compatible with GC9D01)
#define GC9A01_DRIVER

// Set display size
#define TFT_WIDTH  160
#define TFT_HEIGHT 160

// Define pins
#define TFT_MOSI 4
#define TFT_SCLK 14
#define TFT_CS   5    // Will be controlled manually
#define TFT_DC   2
#define TFT_RST  13

// Optional: Set SPI frequency
#define SPI_FREQUENCY  40000000
```

## Servo Bus Connection

The ESP32 is daisy-chained on the Feetech servo bus (half-duplex):
- Connect GPIO19 (RX) to the servo bus data line
- The ESP32 only listens (RX-only operation)
- No TX connection needed for write-only operation

## Protocol

The ESP32 responds to Feetech protocol packets addressed to ID 40:

### Antenna Positions (Register 42-43)
Write 2 bytes to control antenna positions:
- Byte 0: Left antenna (0-255, maps to -1.0 to +1.0)
- Byte 1: Right antenna (0-255, maps to -1.0 to +1.0)

### Eye Mode & Projector (Register 46-47)
Write 2 bytes to control display mode and projector:
- Byte 0: Eye mode (see table below)
- Byte 1: Projector state (0=off, 1=on)

## Eye Modes

| Mode | Name | Description |
|------|------|-------------|
| 0 | Normal | White eyes with automatic blinking |
| 1 | Angry | Red bar eyes with blinking |
| 2 | Heart | Pulsing pink hearts (animated) |
| 3 | Squint | Narrowed horizontal bar |
| 4 | Suspicious | Asymmetric narrowed eyes |
| 5 | Sleepy | Half-closed eyes |

## Building

1. Open `esp32_virtual_servo.ino` in Arduino IDE
2. Select board: "ESP32 Dev Module" or "LOLIN D32"
3. Install and configure TFT_eSPI library (see above)
4. **Hold BOOT button** while uploading
5. Release BOOT after "Connecting..." appears
