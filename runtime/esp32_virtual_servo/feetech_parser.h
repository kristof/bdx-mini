/*
 * Feetech Protocol Parser
 * 
 * Parses Feetech/Dynamixel Protocol 1.0 packets from a serial stream.
 * Only accepts packets addressed to the configured servo ID.
 * 
 * Packet format:
 * [0xFF] [0xFF] [ID] [LENGTH] [INSTRUCTION] [PARAM1] ... [PARAMn] [CHECKSUM]
 * 
 * LENGTH = number of parameters + 2 (instruction + checksum)
 * CHECKSUM = ~(ID + LENGTH + INSTRUCTION + PARAMS...) & 0xFF
 */

#ifndef FEETECH_PARSER_H
#define FEETECH_PARSER_H

#include <Arduino.h>

#define MAX_PACKET_SIZE 64
#define BROADCAST_ID 0xFE

class FeetechParser {
public:
    FeetechParser(uint8_t servoId) : targetId(servoId) {
        reset();
    }
    
    // Process a single byte, returns true when a complete valid packet is received
    bool processByte(uint8_t byte) {
        switch (state) {
            case STATE_HEADER1:
                if (byte == 0xFF) {
                    state = STATE_HEADER2;
                }
                break;
                
            case STATE_HEADER2:
                if (byte == 0xFF) {
                    state = STATE_ID;
                } else {
                    reset();
                }
                break;
                
            case STATE_ID:
                packetId = byte;
                checksum = byte;
                // Only accept packets for our ID or broadcast
                if (packetId == targetId || packetId == BROADCAST_ID) {
                    state = STATE_LENGTH;
                } else {
                    reset();
                }
                break;
                
            case STATE_LENGTH:
                packetLength = byte;
                checksum += byte;
                if (packetLength >= 2 && packetLength < MAX_PACKET_SIZE) {
                    paramIndex = 0;
                    state = STATE_INSTRUCTION;
                } else {
                    reset();
                }
                break;
                
            case STATE_INSTRUCTION:
                instruction = byte;
                checksum += byte;
                expectedParams = packetLength - 2; // -2 for instruction and checksum
                if (expectedParams > 0) {
                    state = STATE_PARAMS;
                } else {
                    state = STATE_CHECKSUM;
                }
                break;
                
            case STATE_PARAMS:
                params[paramIndex++] = byte;
                checksum += byte;
                if (paramIndex >= expectedParams) {
                    state = STATE_CHECKSUM;
                }
                break;
                
            case STATE_CHECKSUM:
                checksum = (~checksum) & 0xFF;
                bool valid = (byte == checksum);
                reset();
                if (valid) {
                    return true;
                }
                break;
        }
        return false;
    }
    
    uint8_t getInstruction() const { return instruction; }
    
    // For WRITE instructions, first param is the start address
    uint8_t getParamStartAddress() const {
        if (expectedParams > 0) {
            return params[0];
        }
        return 0;
    }
    
    // Get pointer to data (params after the address byte)
    uint8_t* getData() { return &params[1]; }
    
    // Get length of data (params minus the address byte)
    uint8_t getDataLength() const {
        return (expectedParams > 1) ? (expectedParams - 1) : 0;
    }
    
private:
    enum State {
        STATE_HEADER1,
        STATE_HEADER2,
        STATE_ID,
        STATE_LENGTH,
        STATE_INSTRUCTION,
        STATE_PARAMS,
        STATE_CHECKSUM
    };
    
    void reset() {
        state = STATE_HEADER1;
        checksum = 0;
        paramIndex = 0;
    }
    
    uint8_t targetId;
    State state = STATE_HEADER1;
    
    uint8_t packetId;
    uint8_t packetLength;
    uint8_t instruction;
    uint8_t params[MAX_PACKET_SIZE];
    uint8_t paramIndex;
    uint8_t expectedParams;
    uint8_t checksum;
};

#endif // FEETECH_PARSER_H
