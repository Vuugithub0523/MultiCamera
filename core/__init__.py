"""
Core module for camera processing pipeline.

Includes: RTSP loading, tracking pipeline, lifecycle management, and multi-camera coordination.
"""
from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "MultiCameraManager": ("core.manager", "MultiCameraManager"),
    "CameraPipeline": ("core.pipeline", "CameraPipeline"),
    "TrackInfo": ("core.pipeline", "TrackInfo"),
    "RTSPStreamLoader": ("core.rtsp_loader", "RTSPStreamLoader"),
    "MultiRTSPLoader": ("core.rtsp_loader", "MultiRTSPLoader"),
    "PersonLifecycleManager": ("core.lifecycle_manager", "PersonLifecycleManager"),
    "PersonLifecycle": ("core.lifecycle_manager", "PersonLifecycle"),
    "PersonState": ("core.lifecycle_manager", "PersonState"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'core' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)


def __dir__():
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))
