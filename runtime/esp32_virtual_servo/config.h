/*
 * Configuration for ESP32 Virtual Servo
 */

#ifndef CONFIG_H
#define CONFIG_H

// Virtual Servo ID on the Feetech bus
#define VIRTUAL_SERVO_ID 40

// Serial communication
#define BAUD_RATE 1000000
#define PIN_SERIAL_RX 16  // RX only, connected to servo bus

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

// Feetech Protocol Constants
#define INST_PING 0x01
#define INST_READ 0x02
#define INST_WRITE 0x03
#define INST_REG_WRITE 0x04
#define INST_ACTION 0x05
#define INST_RESET 0x06
#define INST_SYNC_WRITE 0x83

// Register addresses (STS3215 compatible)
#define REG_GOAL_POSITION 42  // 2 bytes
#define REG_GOAL_SPEED 46     // 2 bytes

// Eye modes
#define EYE_MODE_NORMAL 0
#define EYE_MODE_ANGRY 1
#define EYE_MODE_HEART 2
#define EYE_MODE_SQUINT 3
#define EYE_MODE_SUSPICIOUS 4
#define EYE_MODE_SLEEPY 5
#define EYE_MODE_DIZZY 6

#endif // CONFIG_H
