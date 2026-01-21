"""
Logging utilities
"""
import logging
import sys
from pathlib import Path

from config import Config


def setup_logger(name: str = "native-ai-backend") -> logging.Logger:
    """Setup logger with config"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, Config.LOG_LEVEL))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, Config.LOG_LEVEL))
    
    # Formatter
    formatter = logging.Formatter(Config.LOG_FORMAT)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger
