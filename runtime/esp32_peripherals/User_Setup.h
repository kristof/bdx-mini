//                            USER DEFINED SETTINGS
//   Set driver type, fonts to be loaded, pins used and SPI control method etc.
//
//   FOR ESP32 Virtual Servo - GC9D01 160x160 Dual Display Setup

#define USER_SETUP_INFO "ESP32_Virtual_Servo"

// ##################################################################################
// Section 1. Driver
// ##################################################################################

#define GC9A01_DRIVER

// Display size
#define TFT_WIDTH  160
#define TFT_HEIGHT 160

// Color inversion
#define TFT_INVERSION_OFF

// ##################################################################################
// Section 2. Pins for ESP32
// ##################################################################################

// Color order
#define TFT_RGB_ORDER TFT_BGR

// SPI pins
#define TFT_MOSI 13
#define TFT_SCLK 14

// Control pins (CS is controlled manually for dual display)
// #define TFT_CS   5  // Commented out - controlled manually
#define TFT_DC   2
#define TFT_RST  4

// Backlight
#define TFT_BL   21
#define TFT_BACKLIGHT_ON HIGH

// ##################################################################################
// Section 3. Fonts
// ##################################################################################

#define LOAD_GLCD
#define LOAD_FONT2
#define LOAD_FONT4
#define LOAD_FONT6
#define LOAD_FONT7
#define LOAD_FONT8
#define LOAD_GFXFF
#define SMOOTH_FONT

// ##################################################################################
// Section 4. SPI Frequency
// ##################################################################################

#define SPI_FREQUENCY  20000000
