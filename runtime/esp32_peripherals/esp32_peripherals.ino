/*
 * ESP32 Peripherals
 *
 * Controls the droid's expression hardware:
 * - 2x GC9D01 TFT displays (eyes with 7 expression modes)
 * - 2x PWM servos (antennas)
 * - 1x LED (projector)
 *
 * Two independent serial links carry the same line protocol:
 * - Serial  (native USB): debug output + interactive hotkeys from a laptop
 * - Serial1 (GPIO18 TX / GPIO19 RX, wired to the Pi's hardware UART via the
 *   board's "Serial 0" header): the real command channel from the Pi. This
 *   is a dedicated point-to-point link, independent of the leg-servo bus and
 *   of the USB port above.
 *
 * Eye modes: 0=Normal, 1=Angry, 2=Heart, 3=Squint, 4=Suspicious, 5=Sleepy, 6=Dizzy
 *
 * Line protocol (newline-terminated ASCII):
 *   S,<eye_mode 0-6>,<projector 0|1>,<left_antenna -1.0..1.0>,<right_antenna -1.0..1.0>
 *
 * A handful of single-character hotkeys are also accepted (each followed by
 * Enter) for interactive testing over USB.
 */

#include <Arduino.h>
#include "config.h"
#include "antennas.h"
#include "eyes.h"
#include "projector.h"

// Global instances
Antennas antennas;
Eyes eyes;
Projector projector;

// Current state
uint8_t currentEyeMode = 0;
bool currentProjectorState = false;

#define LINE_BUFFER_SIZE 64

struct LineReader {
    Stream* port;
    char buffer[LINE_BUFFER_SIZE];
    uint8_t length = 0;
};

LineReader usbReader;
LineReader piReader;

void setup() {
    Serial.begin(USB_BAUD_RATE);
    Serial.println("ESP32 Peripherals starting...");

    Serial1.begin(PI_UART_BAUD_RATE, SERIAL_8N1, PIN_PI_UART_RX, PIN_PI_UART_TX);
    Serial.printf("Pi UART initialized (RX=%d, TX=%d) at %d baud\n",
                   PIN_PI_UART_RX, PIN_PI_UART_TX, PI_UART_BAUD_RATE);

    antennas.begin();
    eyes.begin();
    projector.begin();

    Serial.println("All peripherals initialized");

    Serial.println();
    Serial.println("=== Serial Test Commands (press Enter after each) ===");
    Serial.println("  0-6 = Eye modes (0=Normal, 1=Angry, 2=Heart, 3=Squint, 4=Suspicious, 5=Sleepy, 6=Dizzy)");
    Serial.println("  p   = Toggle projector");
    Serial.println("  l   = Left antenna sweep");
    Serial.println("  r   = Right antenna sweep");
    Serial.println("  b   = Both antennas sweep");
    Serial.println("  S,<eye>,<proj>,<left>,<right> = Set full state (used by the Pi)");

    usbReader.port = &Serial;
    piReader.port = &Serial1;
}

void loop() {
    readIncomingLines(usbReader);
    readIncomingLines(piReader);
    eyes.update();
}

void readIncomingLines(LineReader& reader) {
    while (reader.port->available()) {
        char c = reader.port->read();

        if (c == '\n' || c == '\r') {
            if (reader.length > 0) {
                reader.buffer[reader.length] = '\0';
                handleLine(reader.buffer);
                reader.length = 0;
            }
            continue;
        }

        if (reader.length < LINE_BUFFER_SIZE - 1) {
            reader.buffer[reader.length++] = c;
        } else {
            // Line too long, drop it
            reader.length = 0;
        }
    }
}

void handleLine(char* line) {
    if ((line[0] == 'S' || line[0] == 's') && line[1] == ',') {
        handleStateCommand(line);
        return;
    }

    if (line[1] == '\0') {
        handleHotkey(line[0]);
        return;
    }

    Serial.printf("Unknown command: %s\n", line);
}

void handleStateCommand(char* line) {
    int eyeMode;
    int projectorState;
    float leftAntenna;
    float rightAntenna;

    int parsed = sscanf(line, "S,%d,%d,%f,%f", &eyeMode, &projectorState, &leftAntenna, &rightAntenna);
    if (parsed != 4) {
        Serial.printf("Malformed state command: %s\n", line);
        return;
    }

    antennas.setPosition(leftAntenna, rightAntenna);

    if (eyeMode != currentEyeMode && eyeMode >= 0 && eyeMode <= EYE_MODE_DIZZY) {
        currentEyeMode = eyeMode;
        eyes.setMode(currentEyeMode);
    }

    if ((projectorState > 0) != currentProjectorState) {
        currentProjectorState = (projectorState > 0);
        projector.setState(currentProjectorState);
    }
}

void handleHotkey(char cmd) {
    switch (cmd) {
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

        case 'p':
        case 'P':
            currentProjectorState = !currentProjectorState;
            projector.setState(currentProjectorState);
            Serial.printf("Projector: %s\n", currentProjectorState ? "ON" : "OFF");
            break;

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

        default:
            Serial.printf("Unknown command: %c\n", cmd);
            break;
    }
}
