"""
Detection module initialization
"""
from .yolo_detector import YOLODetector
from .byte_tracker import BYTETracker, STrack

__all__ = ['YOLODetector', 'BYTETracker', 'STrack']
