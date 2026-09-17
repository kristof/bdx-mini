"""
ESP32 Peripherals Controller

Controls the ESP32 peripherals board which handles:
- Antenna servos (via PWM)
- Eye displays (GC9D01 TFTs with 7 modes)
- Projector LED

Communicates over the Pi's hardware UART (GPIO14/15, /dev/serial0) - a
dedicated point-to-point link to the ESP32's Serial1 (not the Feetech servo
bus, and not the ESP32's USB port, which is reserved for flashing/debug).
Sends a newline-terminated ASCII line per state update:

    S,<eye_mode>,<projector 0|1>,<left_antenna>,<right_antenna>\\n
"""

import serial


class ESP32Peripherals:
    """
    Controller for ESP32 peripherals over the Pi's UART.
    """

    # Eye modes
    EYE_MODE_NORMAL = 0
    EYE_MODE_ANGRY = 1
    EYE_MODE_HEART = 2
    EYE_MODE_SQUINT = 3
    EYE_MODE_SUSPICIOUS = 4
    EYE_MODE_SLEEPY = 5
    EYE_MODE_DIZZY = 6

    def __init__(self, port: str = "/dev/serial0", baudrate: int = 115200):
        """
        Initialize ESP32 peripherals controller.

        Args:
            port: Serial device for the Pi's UART (GPIO14/15).
            baudrate: Must match PI_UART_BAUD_RATE in the ESP32 firmware's config.h.
        """
        self._port = port
        self._serial = None
        self._connected = False

        try:
            self._serial = serial.Serial(port, baudrate, timeout=0.1)
            self._connected = True
            print(f"ESP32Peripherals: connected on {port}")
        except serial.SerialException as e:
            print(f"ESP32Peripherals: could not open {port} ({e})")

        # Current state tracking
        self._left_antenna = 0.0
        self._right_antenna = 0.0
        self._eye_mode = self.EYE_MODE_NORMAL
        self._projector_on = False

    def _send_state(self) -> bool:
        """Send the full peripheral state as one line over USB serial."""
        if not self._connected:
            return False

        projector_bit = 1 if self._projector_on else 0
        line = (
            f"S,{self._eye_mode},{projector_bit},"
            f"{self._left_antenna:.3f},{self._right_antenna:.3f}\n"
        )

        try:
            self._serial.write(line.encode("ascii"))
            return True
        except serial.SerialException as e:
            print(f"ESP32Peripherals: write failed ({e})")
            return False

    def set_antennas(self, left: float, right: float):
        """
        Set antenna positions.

        Args:
            left: Left antenna position (-1.0 to 1.0)
            right: Right antenna position (-1.0 to 1.0)
        """
        self._left_antenna = max(-1.0, min(1.0, left))
        self._right_antenna = max(-1.0, min(1.0, right))
        self._send_state()

    def set_left_antenna(self, position: float):
        """Set left antenna position (-1.0 to 1.0)."""
        self.set_antennas(position, self._right_antenna)

    def set_right_antenna(self, position: float):
        """Set right antenna position (-1.0 to 1.0)."""
        self.set_antennas(self._left_antenna, position)

    def set_eye_mode(self, mode: int):
        """
        Set eye display mode.

        Args:
            mode: 0=normal, 1=angry, 2=heart, 3=squint, 4=suspicious, 5=sleepy, 6=dizzy
        """
        if mode not in range(7):
            raise ValueError("Eye mode must be 0-6")

        self._eye_mode = mode
        self._send_state()

    def set_projector(self, on: bool):
        """
        Set projector LED state.

        Args:
            on: True to turn on, False to turn off
        """
        self._projector_on = on
        self._send_state()

    def toggle_projector(self):
        """Toggle projector LED state."""
        self.set_projector(not self._projector_on)

    def set_all(self, left_antenna: float, right_antenna: float,
                eye_mode: int, projector_on: bool):
        """
        Set all peripheral states at once.

        Args:
            left_antenna: Left antenna position (-1.0 to 1.0)
            right_antenna: Right antenna position (-1.0 to 1.0)
            eye_mode: 0=normal, 1=angry, 2=heart, 3=squint, 4=suspicious, 5=sleepy, 6=dizzy
            projector_on: True to turn on projector LED
        """
        self._left_antenna = max(-1.0, min(1.0, left_antenna))
        self._right_antenna = max(-1.0, min(1.0, right_antenna))
        self._eye_mode = eye_mode
        self._projector_on = projector_on
        self._send_state()

    def stop(self):
        """Reset all peripherals to default state."""
        self._left_antenna = 0.0
        self._right_antenna = 0.0
        self._eye_mode = self.EYE_MODE_NORMAL
        self._projector_on = False
        self._send_state()

    def close(self):
        """Close the serial connection."""
        if self._serial is not None:
            self._serial.close()

    @property
    def left_antenna(self) -> float:
        return self._left_antenna

    @property
    def right_antenna(self) -> float:
        return self._right_antenna

    @property
    def eye_mode(self) -> int:
        return self._eye_mode

    @property
    def projector_on(self) -> bool:
        return self._projector_on

    @property
    def is_connected(self) -> bool:
        return self._connected


if __name__ == "__main__":
    print("ESP32Peripherals module loaded successfully")
