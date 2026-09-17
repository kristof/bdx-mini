# ESP32 Peripherals

This ESP32 firmware controls the droid's expression hardware over a dedicated
hardware UART link to the Raspberry Pi's GPIO14/15:
- 2x GC9D01 0.71" round TFT displays (eyes with 7 expression modes)
- 2x PWM servos (antennas)
- 1x 3.3V LED (projector)

It used to impersonate a Feetech servo on the leg/head servo bus, but that
bus is half-duplex, shared with 14 time-critical locomotion servos, and only
exposed a single 12-bit register — not enough bandwidth or precision to
emote while walking. It now talks to the Pi over a dedicated point-to-point
UART wire instead, independent of both the servo bus and the ESP32's USB
port (which stays free for flashing/debug).

## Hardware Requirements

- ESP32 D1 Mini (or compatible)
- 2x GC9D01 0.71" round TFT displays (160x160, 8-pin)
- 2x Standard PWM servos (for antennas)
- 1x LED with appropriate resistor (for projector)
- 3 wires from the ESP32's `Serial 0` header (TX/RX/GND) to the Raspberry
  Pi's UART (directly to GPIO14/15, or via a HAT that breaks them out to a
  labeled UART connector) - see Wiring below

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
| Pi UART TX (`Serial1`, drives the board's `Serial 0` TX pin -> Pi RXD) | 18 |
| Pi UART RX (`Serial1`, fed by the board's `Serial 0` RX pin <- Pi TXD) | 19 |

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

## Wiring: Pi UART

Connect 3 wires from the board's `Serial 0` header to the Raspberry Pi's
hardware UART (GPIO14/GPIO15 - either directly on the header, or via a HAT
that breaks them out to a labeled connector):

| `Serial 0` header pin | Connects to |
|---|---|
| TX | Pi UART RX |
| RX | Pi UART TX |
| GND | Pi GND |

(TX/RX cross over, as with any UART link.) On the Pi side this shows up as
`/dev/serial0` - see the main `runtime/README.md` for enabling the Pi's
hardware UART.

This is completely independent of the ESP32's USB port, which stays
available for flashing and debug output/hotkeys the whole time.

## Protocol

The Pi talks to the ESP32 with a newline-terminated ASCII line over the
UART link:

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
