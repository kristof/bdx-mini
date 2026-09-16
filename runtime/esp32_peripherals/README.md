# ESP32 Peripherals

This ESP32 firmware controls the droid's expression hardware over a dedicated
USB serial link to the Raspberry Pi:
- 2x GC9D01 0.71" round TFT displays (eyes with 7 expression modes)
- 2x PWM servos (antennas)
- 1x 3.3V LED (projector)

It used to impersonate a Feetech servo on the leg/head servo bus, but that
bus is half-duplex, shared with 14 time-critical locomotion servos, and only
exposed a single 12-bit register — not enough bandwidth or precision to
emote while walking. It now talks to the Pi over its own USB cable instead.

## Hardware Requirements

- ESP32 D1 Mini (or compatible) with a USB serial port (native USB or
  CP2102/CH340 bridge)
- 2x GC9D01 0.71" round TFT displays (160x160, 8-pin)
- 2x Standard PWM servos (for antennas)
- 1x LED with appropriate resistor (for projector)
- USB cable from the ESP32's USB port to a free USB port on the Raspberry Pi
  (e.g. USB-C on the ESP32 to a spare micro-USB port on the Pi Zero 2W)

## Pin Connections

| Function | GPIO |
|----------|------|
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

## USB Connection

Connect the ESP32 directly to the Raspberry Pi with a USB cable, independent
of the servo bus USB adapter. On the Pi this shows up as its own serial
device (e.g. `/dev/ttyUSB0`), separate from the leg servo bus adapter (e.g.
`/dev/ttyACM0`).

Since there are now two USB-serial devices plugged into the Pi, their
`/dev/ttyUSB*`/`/dev/ttyACM*` numbering isn't guaranteed to stay stable
across reboots or reconnects. Set up a udev rule that maps this ESP32 to a
fixed name (e.g. `/dev/esp32_peripherals`) based on its USB vendor/product ID,
and point the runtime at that path instead of a raw `/dev/ttyUSB0`.

## Protocol

The Pi talks to the ESP32 with a newline-terminated ASCII line over the USB
serial port:

```
S,<eye_mode>,<projector>,<left_antenna>,<right_antenna>
```

- `eye_mode`: integer 0-6 (see table below)
- `projector`: 0 or 1
- `left_antenna`, `right_antenna`: floats from -1.0 to 1.0

Example: `S,2,1,0.50,-0.30\n` sets heart eyes, projector on, left antenna at
0.5, right antenna at -0.3.

A handful of single-character hotkeys (`0`-`6`, `p`, `l`, `r`, `b`, each
followed by Enter) are also accepted for interactive testing from the
Arduino Serial Monitor.

## Eye Modes

| Mode | Name | Description |
|------|------|--------------|
| 0 | Normal | White eyes with automatic blinking |
| 1 | Angry | Red bar eyes with blinking |
| 2 | Heart | Pulsing pink hearts (animated) |
| 3 | Squint | Narrowed horizontal bar |
| 4 | Suspicious | Asymmetric narrowed eyes |
| 5 | Sleepy | Half-closed eyes |
| 6 | Dizzy | Spinning circles |

## Building

1. Open `esp32_peripherals.ino` in Arduino IDE
2. Select board: "ESP32 Dev Module" or "LOLIN D32"
3. Install and configure TFT_eSPI library (see above)
4. **Hold BOOT button** while uploading
5. Release BOOT after "Connecting..." appears
