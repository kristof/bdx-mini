"""
Expressions Module

Coordinates eye modes, antenna positions, and sounds for unified robot expressions.
Each expression bundles visual (eyes, antennas, projector) and audio elements.
"""

import time
import math
from dataclasses import dataclass
from threading import Thread, Event
from typing import Optional, Callable
from mini_bdx_runtime.esp32_peripherals import ESP32Peripherals


@dataclass
class Expression:
    """Defines a complete robot expression."""
    name: str
    eye_mode: int
    left_antenna: float = 0.0         # Static position (used if no animation)
    right_antenna: float = 0.0        # Static position (used if no animation)
    projector: bool = False
    sound: Optional[str] = None
    duration: Optional[float] = None  # How long to hold the expression
    animation: Optional[str] = None   # Default antenna animation pattern
    animation_speed: float = 1.0      # Animation speed multiplier


# Predefined expressions (one per eye mode)
EXPRESSIONS = {
    "neutral": Expression(
        name="neutral",
        eye_mode=ESP32Peripherals.EYE_MODE_NORMAL,
        left_antenna=0.0,
        right_antenna=0.0,
        projector=False,
        sound=None,
        animation=None,  # No animation - static neutral
    ),
    "happy": Expression(
        name="happy",
        eye_mode=ESP32Peripherals.EYE_MODE_HEART,
        left_antenna=0.5,
        right_antenna=0.5,
        projector=False,
        sound="happy.wav",
        animation="bounce",  # Happy bouncing antennas
        animation_speed=1.5,
    ),
    "angry": Expression(
        name="angry",
        eye_mode=ESP32Peripherals.EYE_MODE_ANGRY,
        left_antenna=-0.7,
        right_antenna=-0.7,
        projector=False,
        sound="angry.wav",
        animation="wiggle",  # Tense wiggling
        animation_speed=2.0,
    ),
    "squint": Expression(
        name="squint",
        eye_mode=ESP32Peripherals.EYE_MODE_SQUINT,
        left_antenna=0.0,
        right_antenna=0.0,
        projector=True,
        sound="squint.wav",
        animation=None,  # No animation - focused/static
    ),
    "suspicious": Expression(
        name="suspicious",
        eye_mode=ESP32Peripherals.EYE_MODE_SUSPICIOUS,
        left_antenna=-0.3,
        right_antenna=0.3,
        projector=False,
        sound="suspicious.wav",
        animation="searching",  # Looking around
        animation_speed=0.5,
    ),
    "sleepy": Expression(
        name="sleepy",
        eye_mode=ESP32Peripherals.EYE_MODE_SLEEPY,
        left_antenna=-0.2,
        right_antenna=-0.2,
        projector=False,
        sound="sleepy.wav",
        animation=None,  # No animation - droopy static
    ),
    "dizzy": Expression(
        name="dizzy",
        eye_mode=ESP32Peripherals.EYE_MODE_DIZZY,
        left_antenna=0.0,
        right_antenna=0.0,
        projector=False,
        sound="dizzy.wav",
        animation="alternate",  # Wobbly confusion
        animation_speed=1.5,
    ),
}


class Expressions:
    """
    Manages robot expressions by coordinating eyes, antennas, projector, and sounds.
    
    Usage:
        expressions = Expressions(hwi, sounds)
        expressions.set("happy")
        expressions.set("angry", duration=2.0)
        expressions.reset()
    """
    
    def __init__(self, hwi, sounds=None):
        """
        Initialize expressions manager.
        
        Args:
            hwi: HWI instance with ESP32 peripherals enabled
            sounds: Optional Sounds instance for audio playback
        """
        self.hwi = hwi
        self.sounds = sounds
        self.current_expression = None
        
        # Threaded animation state
        self._animation_thread: Optional[Thread] = None
        self._animation_stop_event = Event()
        self._current_animation: Optional[str] = None
        self._animation_speed: float = 1.0
        
        # Verify ESP32 peripherals are available
        if hwi.esp32 is None:
            raise RuntimeError("HWI must have ESP32 peripherals enabled")
    
    def set(self, expression_name: str, duration: Optional[float] = None, 
            play_sound: bool = True, use_animation: bool = True) -> bool:
        """
        Set a predefined expression.
        
        Args:
            expression_name: Name of the expression (e.g., "happy", "angry")
            duration: Optional duration to hold expression before resetting
            play_sound: Whether to play the associated sound
            use_animation: Whether to use the expression's default animation
            
        Returns:
            True if expression was set successfully
        """
        if expression_name not in EXPRESSIONS:
            print(f"Unknown expression: {expression_name}")
            print(f"Available: {list(EXPRESSIONS.keys())}")
            return False
        
        expr = EXPRESSIONS[expression_name]
        return self._apply_expression(expr, duration, play_sound, use_animation)
    
    def set_custom(self, eye_mode: int, left_antenna: float = 0.0, 
                   right_antenna: float = 0.0, projector: bool = False,
                   sound: Optional[str] = None, duration: Optional[float] = None) -> bool:
        """
        Set a custom expression.
        
        Args:
            eye_mode: Eye mode (0-6)
            left_antenna: Left antenna position (-1.0 to 1.0)
            right_antenna: Right antenna position (-1.0 to 1.0)
            projector: Projector LED state
            sound: Optional sound file to play
            duration: Optional duration before resetting
            
        Returns:
            True if expression was set successfully
        """
        expr = Expression(
            name="custom",
            eye_mode=eye_mode,
            left_antenna=left_antenna,
            right_antenna=right_antenna,
            projector=projector,
            sound=sound,
        )
        return self._apply_expression(expr, duration, True)
    
    def _apply_expression(self, expr: Expression, duration: Optional[float],
                          play_sound: bool, use_animation: bool = True) -> bool:
        """Apply an expression to the robot."""
        try:
            # Stop any existing animation first
            self.stop_antenna_animation(return_to_neutral=False)
            
            # Set eye mode
            self.hwi.set_eye_mode(expr.eye_mode)
            
            # Set projector
            self.hwi.set_projector(expr.projector)
            
            # Handle antennas: use animation if available, otherwise static position
            if use_animation and expr.animation:
                self.start_antenna_animation(expr.animation, expr.animation_speed)
            else:
                self.hwi.set_antennas(expr.left_antenna, expr.right_antenna)
            
            # Play sound if available and requested
            if play_sound and expr.sound and self.sounds:
                self.sounds.play(expr.sound)
            
            self.current_expression = expr
            
            # Hold for duration if specified, then reset
            if duration:
                time.sleep(duration)
                self.reset()
            
            return True
            
        except Exception as e:
            print(f"Error setting expression: {e}")
            return False
    
    def reset(self):
        """Reset to neutral expression (no sound, stops animation)."""
        self.stop_antenna_animation(return_to_neutral=False)
        self.set("neutral", play_sound=False, use_animation=False)
    
    def get_current(self) -> Optional[Expression]:
        """Get the current expression."""
        return self.current_expression
    
    @staticmethod
    def list_expressions() -> list:
        """List all available predefined expressions."""
        return list(EXPRESSIONS.keys())
    
    @staticmethod
    def get_expression(name: str) -> Optional[Expression]:
        """Get an expression definition by name."""
        return EXPRESSIONS.get(name)
    
    def animate_antennas(self, pattern: str = "wave", duration: float = 2.0, 
                         speed: float = 1.0):
        """
        Animate antennas with a pattern (BLOCKING).
        
        Args:
            pattern: Animation pattern ("wave", "bounce", "alternate", "wiggle")
            duration: How long to animate
            speed: Animation speed multiplier
        """
        start_time = time.time()
        
        while time.time() - start_time < duration:
            t = (time.time() - start_time) * speed
            left, right = self._calculate_antenna_position(pattern, t)
            self.hwi.set_antennas(left, right)
            time.sleep(0.02)  # 50Hz update
        
        # Return to neutral
        self.hwi.set_antennas(0.0, 0.0)
    
    def _calculate_antenna_position(self, pattern: str, t: float) -> tuple:
        """Calculate antenna positions for a given pattern and time."""
        if pattern == "wave":
            # Sine wave on both antennas
            val = math.sin(t * 2 * math.pi)
            return val, val
            
        elif pattern == "bounce":
            # Absolute sine (bouncing)
            val = abs(math.sin(t * 2 * math.pi))
            return val, val
            
        elif pattern == "alternate":
            # Opposite directions
            val = math.sin(t * 2 * math.pi)
            return val, -val
            
        elif pattern == "wiggle":
            # Fast small movements
            val = math.sin(t * 6 * math.pi) * 0.3
            return val, val
        
        elif pattern == "excited":
            # Fast alternating bounce
            val = abs(math.sin(t * 4 * math.pi))
            return val, val
        
        elif pattern == "searching":
            # One antenna leads, other follows
            val1 = math.sin(t * 2 * math.pi)
            val2 = math.sin(t * 2 * math.pi - 0.5)
            return val1, val2
        
        elif pattern == "nod":
            # Both antennas nod forward/back
            val = math.sin(t * 3 * math.pi) * 0.5
            return val, val
        
        else:
            return 0.0, 0.0
    
    # ==================== THREADED ANIMATIONS ====================
    
    def start_antenna_animation(self, pattern: str = "wave", speed: float = 1.0):
        """
        Start a non-blocking antenna animation in background thread.
        
        Args:
            pattern: Animation pattern ("wave", "bounce", "alternate", "wiggle", 
                     "excited", "searching", "nod")
            speed: Animation speed multiplier
        """
        # Stop any existing animation
        self.stop_antenna_animation()
        
        self._animation_stop_event.clear()
        self._current_animation = pattern
        self._animation_speed = speed
        
        self._animation_thread = Thread(target=self._animation_loop, daemon=True)
        self._animation_thread.start()
        
        print(f"Started antenna animation: {pattern}")
    
    def stop_antenna_animation(self, return_to_neutral: bool = True):
        """
        Stop the current antenna animation.
        
        Args:
            return_to_neutral: If True, return antennas to neutral position
        """
        if self._animation_thread is not None and self._animation_thread.is_alive():
            self._animation_stop_event.set()
            self._animation_thread.join(timeout=0.5)
            
            if return_to_neutral:
                self.hwi.set_antennas(0.0, 0.0)
            
            print("Stopped antenna animation")
        
        self._current_animation = None
        self._animation_thread = None
    
    def is_animating(self) -> bool:
        """Check if an antenna animation is currently running."""
        return (self._animation_thread is not None and 
                self._animation_thread.is_alive())
    
    def get_current_animation(self) -> Optional[str]:
        """Get the name of the current animation pattern."""
        return self._current_animation if self.is_animating() else None
    
    def _animation_loop(self):
        """Background thread loop for antenna animation."""
        start_time = time.time()
        
        while not self._animation_stop_event.is_set():
            t = (time.time() - start_time) * self._animation_speed
            
            try:
                left, right = self._calculate_antenna_position(
                    self._current_animation, t
                )
                self.hwi.set_antennas(left, right)
            except Exception as e:
                print(f"Animation error: {e}")
                break
            
            # Sleep with check for stop event (allows faster stopping)
            self._animation_stop_event.wait(timeout=0.02)
    
    def set_with_animation(self, expression_name: str, animation: str = "wave",
                           speed: float = 1.0, play_sound: bool = True) -> bool:
        """
        Set an expression with animated antennas (overrides expression's antenna values).
        
        Args:
            expression_name: Name of the expression
            animation: Antenna animation pattern
            speed: Animation speed
            play_sound: Whether to play sound
            
        Returns:
            True if successful
        """
        if expression_name not in EXPRESSIONS:
            print(f"Unknown expression: {expression_name}")
            return False
        
        expr = EXPRESSIONS[expression_name]
        
        # Set eye mode and projector
        self.hwi.set_eye_mode(expr.eye_mode)
        self.hwi.set_projector(expr.projector)
        
        # Play sound
        if play_sound and expr.sound and self.sounds:
            self.sounds.play(expr.sound)
        
        # Start antenna animation
        self.start_antenna_animation(animation, speed)
        
        self.current_expression = expr
        return True
    
    def cleanup(self):
        """Stop all animations and reset. Call before shutting down."""
        self.stop_antenna_animation()
        self.reset()


# Convenience functions for quick access
def create_expression(name: str, eye_mode: int, left_antenna: float = 0.0,
                      right_antenna: float = 0.0, projector: bool = False,
                      sound: Optional[str] = None) -> Expression:
    """Create a custom expression."""
    return Expression(
        name=name,
        eye_mode=eye_mode,
        left_antenna=left_antenna,
        right_antenna=right_antenna,
        projector=projector,
        sound=sound,
    )


def register_expression(expr: Expression):
    """Register a custom expression globally."""
    EXPRESSIONS[expr.name] = expr


if __name__ == "__main__":
    print("Available expressions:")
    for name, expr in EXPRESSIONS.items():
        print(f"  {name}: eye_mode={expr.eye_mode}, "
              f"antennas=({expr.left_antenna}, {expr.right_antenna}), "
              f"projector={expr.projector}, sound={expr.sound}")
