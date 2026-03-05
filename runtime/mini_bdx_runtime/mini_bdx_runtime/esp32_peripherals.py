"""
ESP32 Peripherals Controller

Controls the ESP32 virtual servo (ID 40) which handles:
- Antenna servos (via PWM)
- Eye displays (GC9D01 TFTs with 7 modes)
- Projector LED

Communicates over the Feetech servo bus using rustypot's write_goal_position.
All state is packed into a 12-bit value (0-4095) to fit servo position range.
"""

import math


class ESP32Peripherals:
    """
    Controller for ESP32 virtual servo peripherals.
    
    Packed 12-bit format in goal_position (stays within 0-4095 servo range):
      Bits 0-2:   eye mode (0-6)
      Bit 3:      projector (0 or 1)
      Bits 4-7:   left antenna (0-15, scaled from -1.0 to 1.0)
      Bits 8-11:  right antenna (0-15, scaled from -1.0 to 1.0)
    """
    
    SERVO_ID = 40
    
    # Eye modes
    EYE_MODE_NORMAL = 0
    EYE_MODE_ANGRY = 1
    EYE_MODE_HEART = 2
    EYE_MODE_SQUINT = 3
    EYE_MODE_SUSPICIOUS = 4
    EYE_MODE_SLEEPY = 5
    EYE_MODE_DIZZY = 6
    
    def __init__(self, io):
        """
        Initialize ESP32 peripherals controller.
        
        Args:
            io: rustypot IO instance
        """
        self._io = io
        self._connected = io is not None
        
        # Current state tracking
        self._left_antenna = 0.0
        self._right_antenna = 0.0
        self._eye_mode = self.EYE_MODE_NORMAL
        self._projector_on = False
        
        if self._connected:
            print("ESP32Peripherals: Using rustypot IO for communication")
    
    def _raw_to_radians(self, raw_value: int) -> float:
        """
        Convert a raw 12-bit value to the "radians" value to send to rustypot.
        
        Based on testing: rustypot uses formula ~= radians * 652 + 2048
        To get raw_value N, we need: (N - 2048) / 652 radians
        """
        return (float(raw_value) - 2048.0) / 652.0
    
    def _float_to_4bit(self, value: float) -> int:
        """Convert float (-1.0 to 1.0) to 4-bit value (0-15)."""
        value = max(-1.0, min(1.0, value))
        return int((value + 1.0) * 7.5)
    
    def _send_state(self):
        """
        Send eye mode and projector state via rustypot.
        
        Currently only sends eye mode (bits 0-2) and projector (bit 3).
        Antenna encoding is disabled due to rustypot conversion issues with larger values.
        """
        if not self._connected:
            return False
        
        projector_bit = 1 if self._projector_on else 0
        
        # Only encode eye mode and projector (0-15 range works reliably)
        # Antenna encoding disabled until we solve rustypot conversion for larger values
        raw_value = (self._eye_mode & 0x07) | (projector_bit << 3)
        
        try:
            radians = self._raw_to_radians(raw_value)
            self._io.write_goal_position([self.SERVO_ID], [radians])
            return True
        except Exception:
            return False
    
    def set_antennas(self, left: float, right: float):
        """
        Set antenna positions.
        
        Args:
            left: Left antenna position (-1.0 to 1.0)
            right: Right antenna position (-1.0 to 1.0)
        """
        self._left_antenna = left
        self._right_antenna = right
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
        self._left_antenna = left_antenna
        self._right_antenna = right_antenna
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
