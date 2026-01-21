"""
Core module for camera processing pipeline
Includes: RTSP loading, tracking pipeline, lifecycle management, and multi-camera coordination
"""
from .manager import MultiCameraManager
from .pipeline import CameraPipeline, TrackInfo
from .rtsp_loader import RTSPStreamLoader, MultiRTSPLoader
from .lifecycle_manager import PersonLifecycleManager, PersonLifecycle, PersonState

__all__ = [
    'MultiCameraManager',
    'CameraPipeline',
    'TrackInfo',
    'RTSPStreamLoader',
    'MultiRTSPLoader',
    'PersonLifecycleManager',
    'PersonLifecycle',
    'PersonState',
]
