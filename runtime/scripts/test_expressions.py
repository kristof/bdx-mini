#!/usr/bin/env python3
"""
Test script for robot expressions.

Tests the coordinated expressions (eyes + antennas + sound) using the real
Expressions/ESP32Peripherals classes over the ESP32's USB serial link.
"""

import time
import os
import argparse

from mini_bdx_runtime.expressions import EXPRESSIONS, Expressions
from mini_bdx_runtime.esp32_peripherals import ESP32Peripherals

try:
    from mini_bdx_runtime.sounds import Sounds
    SOUNDS_AVAILABLE = True
except ImportError:
    SOUNDS_AVAILABLE = False


class FakeHWI:
    """Minimal stand-in exposing just the `.esp32` attribute Expressions needs."""

    def __init__(self, esp32: ESP32Peripherals):
        self.esp32 = esp32


def interactive_mode(expressions: Expressions):
    """Interactive expression testing."""
    print("\n=== Interactive Expression Mode ===")
    print("\nAvailable expressions:")
    for i, name in enumerate(EXPRESSIONS.keys()):
        print(f"  {i}: {name}")
    print("\nCommands:")
    print("  <number>    - Set expression by number")
    print("  <name>      - Set expression by name")
    print("  list        - List all expressions")
    print("  reset       - Reset to neutral")
    print("\nThreaded animations:")
    print("  start <pattern> - Start animation")
    print("  stop            - Stop animation")
    print("\n  quit        - Exit")

    expr_list = list(EXPRESSIONS.keys())

    while True:
        try:
            cmd = input("\n> ").strip().lower()
            parts = cmd.split()

            if not parts:
                continue

            if cmd in ("quit", "q"):
                break
            elif cmd == "list":
                for i, name in enumerate(expr_list):
                    print(f"  {i}: {name}")
            elif cmd == "reset":
                expressions.reset()
                print("Reset to neutral")
            elif cmd == "stop":
                expressions.stop_antenna_animation()
            elif parts[0] == "start":
                pattern = parts[1] if len(parts) > 1 else "wave"
                expressions.start_antenna_animation(pattern)
            elif cmd.isdigit():
                idx = int(cmd)
                if 0 <= idx < len(expr_list):
                    name = expr_list[idx]
                    expressions.set(name)
                    print(f"Set: {name}")
                else:
                    print(f"Invalid index. Use 0-{len(expr_list)-1}")
            elif cmd in EXPRESSIONS:
                expressions.set(cmd)
                print(f"Set: {cmd}")
            else:
                print(f"Unknown command: {cmd}")

        except KeyboardInterrupt:
            break

    expressions.cleanup()
    print("\nExiting interactive mode")


def test_all_expressions(expressions: Expressions, delay: float = 3.0):
    """Test all predefined expressions."""
    print("\n=== Testing All Expressions ===\n")
    for name in EXPRESSIONS.keys():
        print(f"Expression: {name}")
        expressions.set(name)
        time.sleep(delay)
    expressions.reset()
    print("\nAll expressions tested!")


def test_antenna_animations(expressions: Expressions):
    """Test antenna animation patterns."""
    print("\n=== Testing Antenna Animations ===\n")
    for pattern in ["wave", "bounce", "alternate", "wiggle"]:
        print(f"Pattern: {pattern}")
        expressions.animate_antennas(pattern, duration=3.0)
        time.sleep(0.5)
    print("\nAnimation tests complete!")


def main():
    parser = argparse.ArgumentParser(description="Test robot expressions")
    parser.add_argument("--port", type=str, default="/dev/serial0",
                        help="Pi UART device connected to the ESP32")
    parser.add_argument("--mode", type=str, default="interactive",
                        choices=["all", "antennas", "interactive"],
                        help="Test mode")
    parser.add_argument("--no-sound", action="store_true",
                        help="Disable sound playback")
    args = parser.parse_args()

    print(f"Connecting to {args.port}...")
    esp32 = ESP32Peripherals(port=args.port)
    if not esp32.is_connected:
        print("Failed to connect to ESP32, exiting.")
        return

    sounds = None
    if not args.no_sound and SOUNDS_AVAILABLE:
        assets_dir = os.path.join(os.path.dirname(__file__),
                                   "../mini_bdx_runtime/assets/")
        if os.path.exists(assets_dir):
            sounds = Sounds(volume=1.0, sound_directory=assets_dir)
            print(f"Sounds loaded from {assets_dir}")
        else:
            print(f"Sound directory not found: {assets_dir}")
    elif not SOUNDS_AVAILABLE:
        print("Sounds module not available")

    expressions = Expressions(FakeHWI(esp32), sounds)
    print(f"Expressions initialized with {len(EXPRESSIONS)} presets")

    try:
        if args.mode == "all":
            test_all_expressions(expressions)
        elif args.mode == "antennas":
            test_antenna_animations(expressions)
        else:
            interactive_mode(expressions)

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print("Resetting...")
        expressions.cleanup()
        esp32.close()
        print("Done")


if __name__ == "__main__":
    main()
