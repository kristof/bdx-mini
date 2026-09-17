/*
 * Configuration for ESP32 Peripherals
 */

#ifndef CONFIG_H
#define CONFIG_H

// Debug/flashing link over the native USB port (no baud negotiation needed
// on boards with native USB CDC; on CP2102/CH340 boards this sets the actual bit rate)
#define USB_BAUD_RATE 115200

// Dedicated hardware UART to the Raspberry Pi's GPIO14/15 (via the board's
// "Serial 0" header), independent of the USB link above.
#define PI_UART_BAUD_RATE 115200
#define PIN_PI_UART_TX 18  // ESP32 transmits here -> Pi RXD (header's TX pin)
#define PIN_PI_UART_RX 19  // ESP32 receives here <- Pi TXD (header's RX pin)

// Antenna PWM pins
#define PIN_ANTENNA_LEFT 25
#define PIN_ANTENNA_RIGHT 26

// TFT Display pins (GC9D01 - matching working User_Setup.h)
#define PIN_TFT_CS_LEFT 15    // Left eye CS (directly controlled)
#define PIN_TFT_CS_RIGHT 5    // Right eye CS (directly controlled)
#define PIN_TFT_DC 2
#define PIN_TFT_CLK 14        // SCLK
#define PIN_TFT_MOSI 13       // MOSI
#define PIN_TFT_RST 4         // Reset
#define PIN_TFT_BL 21         // Backlight

// Projector LED
#define PIN_PROJECTOR 22

// Eye modes
#define EYE_MODE_NORMAL 0
#define EYE_MODE_ANGRY 1
#define EYE_MODE_HEART 2
#define EYE_MODE_SQUINT 3
#define EYE_MODE_SUSPICIOUS 4
#define EYE_MODE_SLEEPY 5
#define EYE_MODE_DIZZY 6

#endif // CONFIG_H
