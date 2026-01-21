"""
Utilities module initialization
"""
from .logger import setup_logger
from .fps_limiter import FpsLimiter

__all__ = ['setup_logger', 'FpsLimiter']
