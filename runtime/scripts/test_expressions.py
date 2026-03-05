#!/usr/bin/env python3
"""
Test script for robot expressions.

Tests the coordinated expressions (eyes + antennas + sound).
"""

import time
import math
import argparse
import os
import serial
from threading import Thread, Event

# Import expressions from the main module (single source of truth)
from mini_bdx_runtime.expressions import EXPRESSIONS, Expression
from mini_bdx_runtime.esp32_peripherals import ESP32Peripherals

# Try to import sounds module
try:
    from mini_bdx_runtime.sounds import Sounds
    SOUNDS_AVAILABLE = True
except ImportError:
    SOUNDS_AVAILABLE = False


class ESP32Controller:
    """Direct ESP32 controller using raw Feetech packets."""
    
    SERVO_ID = 40
    REG_ANTENNAS = 42
    REG_EYE_PROJECTOR = 46
    INST_WRITE = 0x03
    
    def __init__(self, port: str, baudrate: int = 1000000):
        self._serial = serial.Serial(port, baudrate, timeout=0.1)
        self._eye_mode = 0
        self._projector_on = False
    
    def _send_packet(self, register: int, data: list):
        params = [register] + data
        length = len(params) + 2
        packet = [0xFF, 0xFF, self.SERVO_ID, length, self.INST_WRITE] + params
        checksum = (~sum(packet[2:])) & 0xFF
        packet.append(checksum)
        self._serial.write(bytes(packet))
        self._serial.flush()
    
    def _float_to_byte(self, value: float) -> int:
        value = max(-1.0, min(1.0, value))
        return int((value + 1.0) * 127.5)
    
    def set_antennas(self, left: float, right: float):
        left_byte = self._float_to_byte(left)
        right_byte = self._float_to_byte(right)
        self._send_packet(self.REG_ANTENNAS, [left_byte, right_byte])
    
    def set_eye_mode(self, mode: int):
        self._eye_mode = mode
        self._send_packet(self.REG_EYE_PROJECTOR, [self._eye_mode, 1 if self._projector_on else 0])
    
    def set_projector(self, on: bool):
        self._projector_on = on
        self._send_packet(self.REG_EYE_PROJECTOR, [self._eye_mode, 1 if self._projector_on else 0])
    
    def stop(self):
        self.set_antennas(0.0, 0.0)
        self.set_eye_mode(0)
        self.set_projector(False)
    
    def close(self):
        self._serial.close()


class ExpressionsManager:
    """Manages robot expressions with direct ESP32 control."""
    
    def __init__(self, esp32: ESP32Controller, sounds=None):
        self.esp32 = esp32
        self.sounds = sounds
        self.current_expression = None
        self._animation_thread: Optional[Thread] = None
        self._animation_stop_event = Event()
        self._current_animation: Optional[str] = None
        self._animation_speed: float = 1.0
    
    def set(self, expression_name: str, play_sound: bool = True, use_animation: bool = True) -> bool:
        if expression_name not in EXPRESSIONS:
            print(f"Unknown expression: {expression_name}")
            return False
        
        expr = EXPRESSIONS[expression_name]
        
        # Stop any existing animation
        self.stop_antenna_animation(return_to_neutral=False)
        
        # Set eye mode and projector
        self.esp32.set_eye_mode(expr.eye_mode)
        self.esp32.set_projector(expr.projector)
        
        # Handle antennas
        if use_animation and expr.animation:
            self.start_antenna_animation(expr.animation, expr.animation_speed)
        else:
            self.esp32.set_antennas(expr.left_antenna, expr.right_antenna)
        
        # Play sound
        if play_sound and expr.sound and self.sounds:
            self.sounds.play(expr.sound)
        
        self.current_expression = expr
        return True
    
    def reset(self):
        self.stop_antenna_animation(return_to_neutral=False)
        self.set("neutral", play_sound=False, use_animation=False)
    
    def _calculate_antenna_position(self, pattern: str, t: float) -> tuple:
        if pattern == "wave":
            val = math.sin(t * 2 * math.pi)
            return val, val
        elif pattern == "bounce":
            val = abs(math.sin(t * 2 * math.pi))
            return val, val
        elif pattern == "alternate":
            val = math.sin(t * 2 * math.pi)
            return val, -val
        elif pattern == "wiggle":
            val = math.sin(t * 6 * math.pi) * 0.3
            return val, val
        elif pattern == "excited":
            val = abs(math.sin(t * 4 * math.pi))
            return val, val
        elif pattern == "searching":
            val1 = math.sin(t * 2 * math.pi)
            val2 = math.sin(t * 2 * math.pi - 0.5)
            return val1, val2
        elif pattern == "nod":
            val = math.sin(t * 3 * math.pi) * 0.5
            return val, val
        return 0.0, 0.0
    
    def animate_antennas(self, pattern: str = "wave", duration: float = 2.0, speed: float = 1.0):
        """Blocking antenna animation."""
        start_time = time.time()
        while time.time() - start_time < duration:
            t = (time.time() - start_time) * speed
            left, right = self._calculate_antenna_position(pattern, t)
            self.esp32.set_antennas(left, right)
            time.sleep(0.02)
        self.esp32.set_antennas(0.0, 0.0)
    
    def start_antenna_animation(self, pattern: str = "wave", speed: float = 1.0):
        """Start non-blocking antenna animation."""
        self.stop_antenna_animation()
        self._animation_stop_event.clear()
        self._current_animation = pattern
        self._animation_speed = speed
        self._animation_thread = Thread(target=self._animation_loop, daemon=True)
        self._animation_thread.start()
        print(f"Started animation: {pattern}")
    
    def stop_antenna_animation(self, return_to_neutral: bool = True):
        """Stop antenna animation."""
        if self._animation_thread and self._animation_thread.is_alive():
            self._animation_stop_event.set()
            self._animation_thread.join(timeout=0.5)
            if return_to_neutral:
                self.esp32.set_antennas(0.0, 0.0)
            print("Stopped animation")
        self._current_animation = None
        self._animation_thread = None
    
    def _animation_loop(self):
        start_time = time.time()
        while not self._animation_stop_event.is_set():
            t = (time.time() - start_time) * self._animation_speed
            left, right = self._calculate_antenna_position(self._current_animation, t)
            self.esp32.set_antennas(left, right)
            self._animation_stop_event.wait(timeout=0.02)
    
    def set_with_animation(self, expression_name: str, animation: str = "wave",
                           speed: float = 1.0, play_sound: bool = True) -> bool:
        if expression_name not in EXPRESSIONS:
            return False
        expr = EXPRESSIONS[expression_name]
        self.esp32.set_eye_mode(expr.eye_mode)
        self.esp32.set_projector(expr.projector)
        if play_sound and expr.sound and self.sounds:
            self.sounds.play(expr.sound)
        self.start_antenna_animation(animation, speed)
        self.current_expression = expr
        return True
    
    def cleanup(self):
        self.stop_antenna_animation()
        self.reset()


def interactive_mode(expressions: ExpressionsManager):
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
    print("\nBlocking animations (runs for 3s):")
    print("  wave, wiggle, bounce, alternate")
    print("\nThreaded animations:")
    print("  start <pattern> - Start animation")
    print("  stop            - Stop animation")
    print("  combo <expr>    - Expression + wave animation")
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
                expressions.stop_antenna_animation()
                expressions.reset()
                print("Reset to neutral")
            elif cmd == "stop":
                expressions.stop_antenna_animation()
            elif parts[0] == "start":
                pattern = parts[1] if len(parts) > 1 else "wave"
                expressions.start_antenna_animation(pattern)
            elif parts[0] == "combo":
                expr_name = parts[1] if len(parts) > 1 else "happy"
                if expr_name in EXPRESSIONS:
                    expressions.set_with_animation(expr_name, "wave")
                else:
                    print(f"Unknown expression: {expr_name}")
            elif cmd in ("wave", "wiggle", "bounce", "alternate"):
                expressions.animate_antennas(cmd, duration=3.0)
            elif cmd.isdigit():
                idx = int(cmd)
                if 0 <= idx < len(expr_list):
                    name = expr_list[idx]
                    expressions.stop_antenna_animation(return_to_neutral=False)
                    expressions.set(name)
                    print(f"Set: {name}")
                else:
                    print(f"Invalid index. Use 0-{len(expr_list)-1}")
            elif cmd in EXPRESSIONS:
                expressions.stop_antenna_animation(return_to_neutral=False)
                expressions.set(cmd)
                print(f"Set: {cmd}")
            else:
                print(f"Unknown command: {cmd}")
                
        except KeyboardInterrupt:
            break
    
    expressions.cleanup()
    print("\nExiting interactive mode")


def test_all_expressions(expressions: ExpressionsManager, delay: float = 3.0):
    """Test all predefined expressions."""
    print("\n=== Testing All Expressions ===\n")
    for name in EXPRESSIONS.keys():
        print(f"Expression: {name}")
        expressions.set(name)
        time.sleep(delay)
    expressions.reset()
    print("\nAll expressions tested!")


def test_antenna_animations(expressions: ExpressionsManager):
    """Test antenna animation patterns."""
    print("\n=== Testing Antenna Animations ===\n")
    for pattern in ["wave", "bounce", "alternate", "wiggle"]:
        print(f"Pattern: {pattern}")
        expressions.animate_antennas(pattern, duration=3.0)
        time.sleep(0.5)
    print("\nAnimation tests complete!")


def main():
    parser = argparse.ArgumentParser(description="Test robot expressions")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0",
                        help="USB port for servo controller")
    parser.add_argument("--mode", type=str, default="interactive",
                        choices=["all", "antennas", "interactive"],
                        help="Test mode")
    parser.add_argument("--no-sound", action="store_true",
                        help="Disable sound playback")
    args = parser.parse_args()
    
    print(f"Connecting to {args.port}...")
    esp32 = ESP32Controller(args.port)
    print("ESP32 controller initialized")
    
    # Initialize sounds
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
    
    # Create expressions manager
    expressions = ExpressionsManager(esp32, sounds)
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
