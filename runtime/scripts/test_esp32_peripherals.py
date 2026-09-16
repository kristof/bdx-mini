#!/usr/bin/env python3
"""
Test script for ESP32 peripherals (antennas, eyes, projector).

This talks to the ESP32 over its own dedicated USB serial link.
"""

import time
import math
import argparse

from mini_bdx_runtime.esp32_peripherals import ESP32Peripherals


def test_antennas(esp32: ESP32Peripherals, duration: float = 5.0):
    """Test antenna movement with a sine wave pattern."""
    print("\n=== Testing Antennas ===")
    print("Moving antennas in sine wave pattern...")

    start_time = time.monotonic()
    while time.monotonic() - start_time < duration:
        t = time.monotonic() - start_time
        value = math.sin(2 * math.pi * 0.5 * t)  # 0.5 Hz sine wave

        esp32.set_antennas(value, -value)  # Opposite directions
        time.sleep(0.02)  # 50Hz update rate

    # Return to neutral
    esp32.set_antennas(0.0, 0.0)
    print("Antennas test complete")


def test_eyes(esp32: ESP32Peripherals, delay: float = 2.0):
    """Test eye display modes."""
    print("\n=== Testing Eye Modes ===")

    modes = [
        (0, "Normal"),
        (1, "Angry"),
        (2, "Heart"),
        (3, "Squint"),
        (4, "Suspicious"),
        (5, "Sleepy"),
        (6, "Dizzy"),
        (0, "Back to Normal"),
    ]

    for mode, name in modes:
        print(f"Setting eye mode: {name} (mode={mode})")
        esp32.set_eye_mode(mode)
        time.sleep(delay)

    print("Eyes test complete")


def test_projector(esp32: ESP32Peripherals, blink_count: int = 5, delay: float = 0.5):
    """Test projector LED."""
    print("\n=== Testing Projector ===")
    print(f"Blinking projector {blink_count} times...")

    for i in range(blink_count):
        esp32.set_projector(True)
        time.sleep(delay)
        esp32.set_projector(False)
        time.sleep(delay)

    print("Projector test complete")


def test_combined(esp32: ESP32Peripherals, duration: float = 10.0):
    """Test all peripherals together."""
    print("\n=== Combined Test ===")
    print("Testing all peripherals together...")

    start_time = time.monotonic()
    mode_index = 0
    last_mode_change = start_time
    last_projector_toggle = start_time
    projector_on = False

    while time.monotonic() - start_time < duration:
        t = time.monotonic() - start_time

        # Animate antennas
        left_value = math.sin(2 * math.pi * 0.5 * t)
        right_value = math.cos(2 * math.pi * 0.5 * t)
        esp32.set_antennas(left_value, right_value)

        # Change eye mode every 3 seconds
        if time.monotonic() - last_mode_change > 3.0:
            mode_index = (mode_index + 1) % 7
            esp32.set_eye_mode(mode_index)
            last_mode_change = time.monotonic()

        # Toggle projector every 1 second
        if time.monotonic() - last_projector_toggle > 1.0:
            projector_on = not projector_on
            esp32.set_projector(projector_on)
            last_projector_toggle = time.monotonic()

        time.sleep(0.02)

    # Reset to defaults
    esp32.stop()

    print("Combined test complete")


def main():
    parser = argparse.ArgumentParser(description="Test ESP32 peripherals")
    parser.add_argument("--port", type=str, default="/dev/esp32_peripherals",
                        help="USB serial port for the ESP32")
    parser.add_argument("--test", type=str, default="all",
                        choices=["antennas", "eyes", "projector", "combined", "all"],
                        help="Which test to run")
    args = parser.parse_args()

    print(f"Connecting to {args.port}...")
    esp32 = ESP32Peripherals(port=args.port)

    if not esp32.is_connected:
        print("Failed to connect to ESP32, exiting.")
        return

    try:
        if args.test in ("antennas", "all"):
            test_antennas(esp32)

        if args.test in ("eyes", "all"):
            test_eyes(esp32)

        if args.test in ("projector", "all"):
            test_projector(esp32)

        if args.test in ("combined", "all"):
            test_combined(esp32)

        print("\n=== All tests complete ===")

    except KeyboardInterrupt:
        print("\nTest interrupted")
    finally:
        print("Resetting peripherals...")
        esp32.stop()
        esp32.close()
        print("Done")


if __name__ == "__main__":
    main()
