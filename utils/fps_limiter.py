"""
FPS Limiter utility
"""
import time


class FpsLimiter:
    """Rate limiter for controlling FPS"""
    
    def __init__(self, target_fps: float):
        """
        Initialize FPS limiter
        
        Args:
            target_fps: Target frames per second
        """
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps if target_fps > 0 else 0
        self.last_time = time.time()
    
    def wait(self):
        """Wait to maintain target FPS"""
        if self.frame_interval <= 0:
            return
        
        now = time.time()
        elapsed = now - self.last_time
        sleep_time = self.frame_interval - elapsed
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        self.last_time = time.time()
    
    def reset(self):
        """Reset timer"""
        self.last_time = time.time()
