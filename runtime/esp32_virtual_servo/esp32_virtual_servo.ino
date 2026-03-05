/*
 * ESP32 Virtual Servo - ID 40
 * 
 * Acts as a virtual Feetech servo on the serial bus, controlling:
 * - 2x GC9D01 TFT displays (eyes with 7 expression modes)
 * - 2x PWM servos (antennas)
 * - 1x LED (projector)
 * 
 * Eye modes: 0=Normal, 1=Angry, 2=Heart, 3=Squint, 4=Suspicious, 5=Sleepy, 6=Dizzy
 * 
 * Listens for Feetech protocol packets addressed to ID 40.
 * 
 * Goal Position (reg 42-43) packed 12-bit format (0-4095):
 *   Bits 0-2:   eye mode (0-6)
 *   Bit 3:      projector (0 or 1)
 *   Bits 4-7:   left antenna (0-15)
 *   Bits 8-11:  right antenna (0-15)
 */

#include <Arduino.h>
#include "config.h"
#include "feetech_parser.h"
#include "antennas.h"
#include "eyes.h"
#include "projector.h"

// Global instances
FeetechParser parser(VIRTUAL_SERVO_ID);
Antennas antennas;
Eyes eyes;
Projector projector;

// Current state
uint8_t currentEyeMode = 0;
bool currentProjectorState = false;
uint8_t leftAntennaPos = 128;  // Neutral position
uint8_t rightAntennaPos = 128; // Neutral position

void setup() {
    // Initialize Serial for debugging (USB)
    Serial.begin(115200);
    Serial.println("ESP32 Virtual Servo starting...");
    
    // Initialize Serial1 for Feetech bus (RX only)
    Serial1.begin(BAUD_RATE, SERIAL_8N1, PIN_SERIAL_RX, -1);
    Serial.printf("Feetech bus initialized on GPIO%d at %d baud\n", PIN_SERIAL_RX, BAUD_RATE);
    
    // Initialize peripherals
    antennas.begin();
    eyes.begin();
    projector.begin();
    
    Serial.println("All peripherals initialized");
    Serial.printf("Listening for packets addressed to ID %d\n", VIRTUAL_SERVO_ID);
    
    // Print serial commands help
    Serial.println();
    Serial.println("=== Serial Test Commands ===");
    Serial.println("  0-6 = Eye modes (0=Normal, 1=Angry, 2=Heart, 3=Squint, 4=Suspicious, 5=Sleepy, 6=Dizzy)");
    Serial.println("  p   = Toggle projector");
    Serial.println("  l   = Left antenna sweep");
    Serial.println("  r   = Right antenna sweep");
    Serial.println("  b   = Both antennas sweep");
}

void loop() {
    // Read incoming bytes from servo bus
    while (Serial1.available()) {
        uint8_t byte = Serial1.read();
        
        if (parser.processByte(byte)) {
            // Complete packet received for our ID
            handlePacket();
        }
    }
    
    // Handle serial commands for testing
    handleSerialCommands();
    
    // Update eyes animation (handles blinking etc.)
    eyes.update();
}

void handleSerialCommands() {
    if (Serial.available()) {
        char cmd = Serial.read();
        
        switch (cmd) {
            // Eye modes 0-5
            case '0':
                eyes.setMode(EYE_MODE_NORMAL);
                currentEyeMode = EYE_MODE_NORMAL;
                Serial.println("Eye mode: Normal");
                break;
            case '1':
                eyes.setMode(EYE_MODE_ANGRY);
                currentEyeMode = EYE_MODE_ANGRY;
                Serial.println("Eye mode: Angry");
                break;
            case '2':
                eyes.setMode(EYE_MODE_HEART);
                currentEyeMode = EYE_MODE_HEART;
                Serial.println("Eye mode: Heart");
                break;
            case '3':
                eyes.setMode(EYE_MODE_SQUINT);
                currentEyeMode = EYE_MODE_SQUINT;
                Serial.println("Eye mode: Squint");
                break;
            case '4':
                eyes.setMode(EYE_MODE_SUSPICIOUS);
                currentEyeMode = EYE_MODE_SUSPICIOUS;
                Serial.println("Eye mode: Suspicious");
                break;
            case '5':
                eyes.setMode(EYE_MODE_SLEEPY);
                currentEyeMode = EYE_MODE_SLEEPY;
                Serial.println("Eye mode: Sleepy");
                break;
            case '6':
                eyes.setMode(EYE_MODE_DIZZY);
                currentEyeMode = EYE_MODE_DIZZY;
                Serial.println("Eye mode: Dizzy");
                break;
            
            // Projector toggle
            case 'p':
            case 'P':
                currentProjectorState = !currentProjectorState;
                projector.setState(currentProjectorState);
                Serial.printf("Projector: %s\n", currentProjectorState ? "ON" : "OFF");
                break;
            
            // Antenna tests
            case 'l':
            case 'L':
                Serial.println("Left antenna sweep...");
                for (float v = -1.0; v <= 1.0; v += 0.1) {
                    antennas.setLeft(v);
                    delay(50);
                }
                antennas.setLeft(0);
                Serial.println("Done");
                break;
            
            case 'r':
            case 'R':
                Serial.println("Right antenna sweep...");
                for (float v = -1.0; v <= 1.0; v += 0.1) {
                    antennas.setRight(v);
                    delay(50);
                }
                antennas.setRight(0);
                Serial.println("Done");
                break;
            
            case 'b':
            case 'B':
                Serial.println("Both antennas sweep...");
                for (float v = -1.0; v <= 1.0; v += 0.1) {
                    antennas.setPosition(v, -v);
                    delay(50);
                }
                antennas.setPosition(0, 0);
                Serial.println("Done");
                break;
        }
    }
}

void handlePacket() {
    uint8_t instruction = parser.getInstruction();
    uint8_t startAddr;
    uint8_t* data;
    uint8_t dataLen;
    
    if (instruction == INST_WRITE) {
        // Regular WRITE: [addr] [data...]
        startAddr = parser.getParamStartAddress();
        data = parser.getData();
        dataLen = parser.getDataLength();
    }
    else if (instruction == INST_SYNC_WRITE) {
        // SYNC_WRITE: [addr] [len] [id1] [data1...] [id2] [data2...] ...
        uint8_t* params = parser.getData() - 1;  // Include start address
        uint8_t totalParams = parser.getDataLength() + 1;
        
        if (totalParams < 4) return;  // Need at least addr, len, id, data
        
        startAddr = params[0];
        uint8_t dataLenPerServo = params[1];
        
        // Search for our ID in the sync write
        uint8_t offset = 2;  // Start after addr and len
        bool found = false;
        
        while (offset + dataLenPerServo < totalParams) {
            uint8_t servoId = params[offset];
            if (servoId == VIRTUAL_SERVO_ID) {
                data = &params[offset + 1];
                dataLen = dataLenPerServo;
                found = true;
                break;
            }
            offset += 1 + dataLenPerServo;  // Skip to next servo
        }
        
        if (!found) return;  // Our ID not in this sync write
    }
    else {
        return;  // Unknown instruction
    }
    
    
    // Handle Goal Position register (address 42-43)
    // Format: packed 12-bit value (0-4095) containing all state
    //   Bits 0-2:   eye mode (0-6)
    //   Bit 3:      projector (0 or 1)
    //   Bits 4-7:   left antenna (0-15)
    //   Bits 8-11:  right antenna (0-15)
    if (startAddr == REG_GOAL_POSITION && dataLen >= 2) {
        uint16_t rawValue = data[0] | (data[1] << 8);
        
        // Decode packed format
        uint8_t eyeMode = rawValue & 0x07;           // Bits 0-2
        uint8_t projectorState = (rawValue >> 3) & 0x01;  // Bit 3
        uint8_t left4bit = (rawValue >> 4) & 0x0F;   // Bits 4-7
        uint8_t right4bit = (rawValue >> 8) & 0x0F;  // Bits 8-11
        
        // Scale 4-bit values (0-15) to full range (0-255)
        leftAntennaPos = left4bit * 17;   // 0-15 -> 0-255
        rightAntennaPos = right4bit * 17;
        
        // Set antenna positions (-1.0 to 1.0)
        float leftVal = (leftAntennaPos / 127.5f) - 1.0f;
        float rightVal = (rightAntennaPos / 127.5f) - 1.0f;
        antennas.setPosition(leftVal, rightVal);
        
        // Set eye mode
        if (eyeMode != currentEyeMode && eyeMode <= EYE_MODE_DIZZY) {
            currentEyeMode = eyeMode;
            eyes.setMode(currentEyeMode);
        }
        
        // Set projector
        if ((projectorState > 0) != currentProjectorState) {
            currentProjectorState = (projectorState > 0);
            projector.setState(currentProjectorState);
        }
    }
    
    // Legacy: Handle Goal Speed register (address 46-47) for direct test commands
    if (startAddr == REG_GOAL_SPEED && dataLen >= 2) {
        uint8_t eyeMode = data[0];
        uint8_t projectorState = data[1];
        
        if (eyeMode != currentEyeMode && eyeMode <= EYE_MODE_DIZZY) {
            currentEyeMode = eyeMode;
            eyes.setMode(currentEyeMode);
            Serial.printf("Eye mode (legacy): %d\n", currentEyeMode);
        }
        
        if ((projectorState > 0) != currentProjectorState) {
            currentProjectorState = (projectorState > 0);
            projector.setState(currentProjectorState);
            Serial.printf("Projector (legacy): %s\n", currentProjectorState ? "ON" : "OFF");
        }
    }
}
