/*
 * Antenna PWM Servo Control
 * 
 * Controls two standard PWM servos for the antennas.
 * Uses ESP32 LEDC peripheral for hardware PWM.
 * 
 * Compatible with ESP32 Arduino Core 3.x
 */

#ifndef ANTENNAS_H
#define ANTENNAS_H

#include <Arduino.h>
#include "config.h"

// PWM configuration for servos
#define SERVO_FREQ 50        // 50Hz = 20ms period
#define SERVO_RESOLUTION 16  // 16-bit resolution
#define SERVO_MIN_US 1000    // 1ms pulse = -1.0 position
#define SERVO_MAX_US 2000    // 2ms pulse = +1.0 position
#define SERVO_NEUTRAL_US 1500 // 1.5ms pulse = 0.0 position

class Antennas {
public:
    void begin() {
        // Configure LEDC for left antenna (ESP32 Arduino Core 3.x API)
        ledcAttach(PIN_ANTENNA_LEFT, SERVO_FREQ, SERVO_RESOLUTION);
        
        // Configure LEDC for right antenna
        ledcAttach(PIN_ANTENNA_RIGHT, SERVO_FREQ, SERVO_RESOLUTION);
        
        // Set to neutral position
        setPosition(0.0f, 0.0f);
        
        Serial.println("Antennas initialized");
    }
    
    // Set antenna positions (-1.0 to 1.0)
    void setPosition(float left, float right) {
        setServo(PIN_ANTENNA_LEFT, left);
        setServo(PIN_ANTENNA_RIGHT, right);
    }
    
    void setLeft(float position) {
        setServo(PIN_ANTENNA_LEFT, position);
    }
    
    void setRight(float position) {
        setServo(PIN_ANTENNA_RIGHT, position);
    }
    
private:
    void setServo(uint8_t pin, float position) {
        // Clamp position to valid range
        position = constrain(position, -1.0f, 1.0f);
        
        // Map -1.0...1.0 to pulse width in microseconds
        uint32_t pulseUs = SERVO_NEUTRAL_US + (int32_t)(position * 500.0f);
        
        // Convert microseconds to duty cycle
        // At 50Hz, period = 20000us
        // duty = (pulseUs / 20000) * (2^16 - 1)
        uint32_t duty = (pulseUs * 65535) / 20000;
        
        // ESP32 Arduino Core 3.x uses pin instead of channel
        ledcWrite(pin, duty);
    }
};

#endif // ANTENNAS_H
