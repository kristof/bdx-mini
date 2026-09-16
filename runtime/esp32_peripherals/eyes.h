/*
 * Eyes Display Control - TFT_eSPI Version
 * 
 * Controls two GC9D01 160x160 TFT displays for eye expressions.
 * Uses TFT_eSPI library with dual CS pin control.
 * 
 * Modes:
 * 0 - Normal (white eyes with blinking)
 * 1 - Angry (red angled eyes)
 * 2 - Heart (pulsing hearts)
 * 3 - Squint (narrow horizontal)
 * 4 - Suspicious (looking side to side)
 * 5 - Sleepy (half-closed)
 * 6 - Dizzy (spinning circles)
 */

#ifndef EYES_H
#define EYES_H

#include <Arduino.h>
#include <TFT_eSPI.h>
#include <SPI.h>
#include "config.h"

// Colors
#define EYE_COLOR   TFT_WHITE
#define BG_COLOR    TFT_BLACK

class Eyes {
public:
    void begin() {
        Serial.println("Initializing eyes...");
        
        // Setup CS pins FIRST
        pinMode(PIN_TFT_CS_LEFT, OUTPUT);
        pinMode(PIN_TFT_CS_RIGHT, OUTPUT);
        deselectAll();
        
        // Setup backlight
        pinMode(PIN_TFT_BL, OUTPUT);
        digitalWrite(PIN_TFT_BL, HIGH);
        
        // Initialize BOTH displays at once (same as working code)
        selectBoth();
        tft.init();
        tft.setRotation(0);
        tft.fillScreen(BG_COLOR);
        deselectAll();
        
        delay(200);
        
        // Run boot sequence
        bootSequence();
        
        // Set initial state
        currentMode = EYE_MODE_NORMAL;
        isAngry = false;
        heartMode = false;
        autoBlinkEnabled = true;
        nextBlinkTime = millis() + random(2000, 5000);
        
        Serial.println("Eyes initialized");
    }
    
    void setMode(uint8_t mode) {
        if (mode == currentMode) return;
        
        currentMode = mode;
        
        switch (mode) {
            case EYE_MODE_NORMAL:
                eyesOpen();
                isAngry = false;
                heartMode = false;
                squintMode = false;
                dizzyMode = false;
                autoBlinkEnabled = true;
                break;
            case EYE_MODE_ANGRY:
                angryEyes();
                isAngry = true;
                heartMode = false;
                squintMode = false;
                dizzyMode = false;
                autoBlinkEnabled = true;
                break;
            case EYE_MODE_HEART:
                heartMode = true;
                squintMode = false;
                dizzyMode = false;
                isAngry = false;
                autoBlinkEnabled = false;
                drawHeart(35);
                break;
            case EYE_MODE_SQUINT:
                squintEyes();
                isAngry = false;
                heartMode = false;
                squintMode = false;
                dizzyMode = false;
                autoBlinkEnabled = false;
                break;
            case EYE_MODE_SUSPICIOUS:
                squintMode = true;  // Suspicious uses the animated looking around
                isAngry = false;
                heartMode = false;
                dizzyMode = false;
                autoBlinkEnabled = false;
                suspiciousLookingLeft = true;
                drawSuspiciousLook(true);  // Start looking left
                break;
            case EYE_MODE_SLEEPY:
                sleepy();
                isAngry = false;
                heartMode = false;
                squintMode = false;
                dizzyMode = false;
                autoBlinkEnabled = false;
                break;
            case EYE_MODE_DIZZY:
                dizzyMode = true;
                isAngry = false;
                heartMode = false;
                squintMode = false;
                autoBlinkEnabled = false;
                dizzyFrame = 0;
                dizzyLoops = 0;
                drawDizzyFrame(0);
                break;
            default:
                eyesOpen();
                isAngry = false;
                heartMode = false;
                squintMode = false;
                dizzyMode = false;
                autoBlinkEnabled = true;
                break;
        }
    }
    
    void update() {
        // Heart mode animation
        if (heartMode) {
            updateHeartPulse();
            return;
        }
        
        // Suspicious mode animation (looking side to side)
        if (squintMode) {
            updateSuspicious();
            return;
        }
        
        // Dizzy mode animation (spinning circles)
        if (dizzyMode) {
            updateDizzy();
            return;
        }
        
        // Auto-blink
        if (autoBlinkEnabled && (millis() >= nextBlinkTime)) {
            nextBlinkTime = millis() + random(2000, 5000);
            
            if (isAngry) {
                blinkAngry();
            } else {
                blinkBothEyes();
            }
        }
    }
    
    void updateSuspicious() {
        if (millis() - lastSuspicious > 800) {  // Jump every 800ms
            lastSuspicious = millis();
            suspiciousLookingLeft = !suspiciousLookingLeft;
            drawSuspiciousLook(suspiciousLookingLeft);
        }
    }
    
    void drawSuspiciousLook(bool lookLeft) {
        if (lookLeft) {
            // Looking left: left eye wide, right eye narrow
            selectLeft();
            tft.fillScreen(BG_COLOR);
            tft.fillRect(0, 40, 160, 80, EYE_COLOR);
            
            selectRight();
            tft.fillScreen(BG_COLOR);
            tft.fillRect(0, 60, 160, 40, EYE_COLOR);
        } else {
            // Looking right: right eye wide, left eye narrow
            selectLeft();
            tft.fillScreen(BG_COLOR);
            tft.fillRect(0, 60, 160, 40, EYE_COLOR);
            
            selectRight();
            tft.fillScreen(BG_COLOR);
            tft.fillRect(0, 40, 160, 80, EYE_COLOR);
        }
        
        deselectAll();
        drawScanlines();
    }
    
    void updateDizzy() {
        if (millis() - lastDizzy > 100) {  // Update every 100ms
            lastDizzy = millis();
            
            dizzyFrame++;
            if (dizzyFrame >= 4) {
                dizzyFrame = 0;
            }
            
            drawDizzyFrame(dizzyFrame);
        }
    }
    
    void drawDizzyFrame(uint8_t frame) {
        // Circle positions for spinning effect
        int x, y;
        switch (frame) {
            case 0: x = 50;  y = 80;  break;  // Left
            case 1: x = 80;  y = 50;  break;  // Top
            case 2: x = 110; y = 80;  break;  // Right
            case 3: x = 80;  y = 110; break;  // Bottom
            default: x = 80; y = 80; break;
        }
        
        selectBoth();
        tft.fillScreen(BG_COLOR);
        tft.fillCircle(x, y, 30, EYE_COLOR);
        deselectAll();
        drawScanlines();
    }
    
private:
    TFT_eSPI tft = TFT_eSPI();
    
    uint8_t currentMode = EYE_MODE_NORMAL;
    bool isAngry = false;
    bool heartMode = false;
    bool squintMode = false;
    bool autoBlinkEnabled = true;
    uint32_t nextBlinkTime = 0;
    
    // Heart animation state
    int heartSize = 35;
    int heartDir = 1;
    uint32_t lastPulse = 0;
    
    // Suspicious animation state
    bool suspiciousLookingLeft = true;
    uint32_t lastSuspicious = 0;
    
    // Dizzy animation state
    bool dizzyMode = false;
    uint8_t dizzyFrame = 0;
    uint8_t dizzyLoops = 0;
    uint32_t lastDizzy = 0;
    
    //--------------------------------------------------
    // Screen selection
    //--------------------------------------------------
    void selectLeft() {
        digitalWrite(PIN_TFT_CS_RIGHT, HIGH);
        digitalWrite(PIN_TFT_CS_LEFT, LOW);
    }
    
    void selectRight() {
        digitalWrite(PIN_TFT_CS_LEFT, HIGH);
        digitalWrite(PIN_TFT_CS_RIGHT, LOW);
    }
    
    void selectBoth() {
        digitalWrite(PIN_TFT_CS_LEFT, LOW);
        digitalWrite(PIN_TFT_CS_RIGHT, LOW);
    }
    
    void deselectAll() {
        digitalWrite(PIN_TFT_CS_LEFT, HIGH);
        digitalWrite(PIN_TFT_CS_RIGHT, HIGH);
    }
    
    //--------------------------------------------------
    // Scanlines overlay - adds droid CRT effect
    //--------------------------------------------------
    void drawScanlines() {
        selectBoth();
        for (int y = 0; y < 160; y += 4) {
            tft.drawFastHLine(0, y, 160, BG_COLOR);
            tft.drawFastHLine(0, y + 1, 160, BG_COLOR);
        }
        deselectAll();
    }
    
    void drawScanlinesLeft() {
        selectLeft();
        for (int y = 0; y < 160; y += 4) {
            tft.drawFastHLine(0, y, 160, BG_COLOR);
            tft.drawFastHLine(0, y + 1, 160, BG_COLOR);
        }
        deselectAll();
    }
    
    void drawScanlinesRight() {
        selectRight();
        for (int y = 0; y < 160; y += 4) {
            tft.drawFastHLine(0, y, 160, BG_COLOR);
            tft.drawFastHLine(0, y + 1, 160, BG_COLOR);
        }
        deselectAll();
    }
    
    //--------------------------------------------------
    // Eye states
    //--------------------------------------------------
    void eyesOpen() {
        selectBoth();
        tft.fillScreen(EYE_COLOR);
        deselectAll();
        drawScanlines();
    }
    
    void eyesClosed() {
        selectBoth();
        tft.fillScreen(BG_COLOR);
        deselectAll();
    }
    
    void angryEyes() {
        // Left eye - triangle slopes down toward center (inner edge lower)
        // Inner edge is on the RIGHT side of left eye
        selectLeft();
        tft.fillScreen(BG_COLOR);
        tft.fillTriangle(
            160, 30,    // Top right (outer edge - high)
            0, 80,      // Top left (inner edge - lower)
            0, 130,     // Bottom left
            TFT_RED
        );
        tft.fillTriangle(
            160, 30,    // Top right
            160, 130,   // Bottom right
            0, 130,     // Bottom left
            TFT_RED
        );
        
        // Right eye - triangle slopes down toward center (inner edge lower)
        // Inner edge is on the LEFT side of right eye
        selectRight();
        tft.fillScreen(BG_COLOR);
        tft.fillTriangle(
            0, 30,      // Top left (outer edge - high)
            160, 80,    // Top right (inner edge - lower)
            160, 130,   // Bottom right
            TFT_RED
        );
        tft.fillTriangle(
            0, 30,      // Top left
            0, 130,     // Bottom left
            160, 130,   // Bottom right
            TFT_RED
        );
        
        deselectAll();
        drawScanlines();
    }
    
    //--------------------------------------------------
    // Heart eyes
    //--------------------------------------------------
    void drawHeart(int size) {
        selectBoth();
        tft.fillScreen(BG_COLOR);
        
        int cx = 80;
        int cy = 65;
        uint16_t hotPink = tft.color565(255, 20, 147);
        
        // Two circles for bumps
        tft.fillCircle(cx - size/2, cy, size/2, hotPink);
        tft.fillCircle(cx + size/2, cy, size/2, hotPink);
        
        // Triangle for bottom point
        tft.fillTriangle(cx - size, cy, cx + size, cy, cx, cy + size + 10, hotPink);
        
        // Rectangle to fill gap
        tft.fillRect(cx - size/2, cy, size, size/2, hotPink);
        
        // Scanlines
        for (int y = 0; y < 160; y += 4) {
            tft.drawFastHLine(0, y, 160, BG_COLOR);
            tft.drawFastHLine(0, y + 1, 160, BG_COLOR);
        }
        
        deselectAll();
    }
    
    void updateHeartPulse() {
        if (millis() - lastPulse > 60) {
            lastPulse = millis();
            
            heartSize += heartDir * 2;
            if (heartSize >= 50) heartDir = -1;
            if (heartSize <= 30) heartDir = 1;
            
            drawHeart(heartSize);
        }
    }
    
    //--------------------------------------------------
    // Blink animations
    //--------------------------------------------------
    void blinkBothEyes() {
        // Close - eyelids come together
        for (int i = 0; i <= 80; i += 8) {
            selectBoth();
            tft.fillScreen(EYE_COLOR);
            tft.fillRect(0, 0, 160, i, BG_COLOR);
            tft.fillRect(0, 160 - i, 160, i, BG_COLOR);
            // Scanlines on visible area
            for (int y = i; y < 160 - i; y += 4) {
                tft.drawFastHLine(0, y, 160, BG_COLOR);
                tft.drawFastHLine(0, y + 1, 160, BG_COLOR);
            }
            deselectAll();
            delay(10);
        }
        
        eyesClosed();
        delay(80);
        
        // Open
        for (int i = 80; i >= 0; i -= 8) {
            selectBoth();
            tft.fillScreen(EYE_COLOR);
            tft.fillRect(0, 0, 160, i, BG_COLOR);
            tft.fillRect(0, 160 - i, 160, i, BG_COLOR);
            // Scanlines on visible area
            for (int y = i; y < 160 - i; y += 4) {
                tft.drawFastHLine(0, y, 160, BG_COLOR);
                tft.drawFastHLine(0, y + 1, 160, BG_COLOR);
            }
            deselectAll();
            delay(10);
        }
        
        eyesOpen();
    }
    
    void blinkAngry() {
        selectBoth();
        tft.fillScreen(BG_COLOR);
        deselectAll();
        delay(80);
        angryEyes();
    }
    
    void blinkFast() {
        eyesClosed();
        delay(60);
        if (isAngry) {
            angryEyes();
        } else {
            eyesOpen();
        }
    }
    
    //--------------------------------------------------
    // Expressions
    //--------------------------------------------------
    void squintEyes() {
        selectBoth();
        tft.fillScreen(BG_COLOR);
        tft.fillRect(0, 55, 160, 50, EYE_COLOR);
        deselectAll();
        drawScanlines();
    }
    
    void suspicious() {
        selectLeft();
        tft.fillScreen(BG_COLOR);
        tft.fillRect(0, 60, 160, 40, EYE_COLOR);
        
        selectRight();
        tft.fillScreen(BG_COLOR);
        tft.fillRect(0, 40, 160, 80, EYE_COLOR);
        
        deselectAll();
        drawScanlines();
    }
    
    void sleepy() {
        selectBoth();
        tft.fillScreen(BG_COLOR);
        tft.fillRect(0, 80, 160, 40, EYE_COLOR);
        deselectAll();
        drawScanlines();
    }
    
    //--------------------------------------------------
    // Boot sequence
    //--------------------------------------------------
    void bootSequence() {
        Serial.println("Booting eyes...");
        
        eyesClosed();
        delay(500);
        
        // Scanline boot effect
        selectBoth();
        for (int y = 0; y < 160; y += 2) {
            tft.drawFastHLine(0, y, 160, EYE_COLOR);
            delay(10);
        }
        deselectAll();
        delay(100);
        
        eyesOpen(); delay(50);
        eyesClosed(); delay(100);
        eyesOpen(); delay(30);
        eyesClosed(); delay(80);
        eyesOpen();
        
        delay(300);
        
        blinkFast();
        delay(200);
        blinkFast();
        
        Serial.println("Eyes online.");
    }
};

#endif // EYES_H
