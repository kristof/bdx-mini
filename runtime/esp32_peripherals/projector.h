/*
 * Projector LED Control
 * 
 * Simple GPIO control for the 3.3V projector LED.
 */

#ifndef PROJECTOR_H
#define PROJECTOR_H

#include <Arduino.h>
#include "config.h"

class Projector {
public:
    void begin() {
        pinMode(PIN_PROJECTOR, OUTPUT);
        digitalWrite(PIN_PROJECTOR, LOW);
        state = false;
        Serial.println("Projector initialized");
    }
    
    void setState(bool on) {
        state = on;
        digitalWrite(PIN_PROJECTOR, state ? HIGH : LOW);
    }
    
    void toggle() {
        setState(!state);
    }
    
    bool getState() const {
        return state;
    }
    
private:
    bool state = false;
};

#endif // PROJECTOR_H
