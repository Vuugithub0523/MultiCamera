"""
API Models (Pydantic schemas)
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class CameraInfo(BaseModel):
    """Camera information"""
    id: str
    name: str
    rtsp_url: str
    enabled: bool
    

class CameraStats(BaseModel):
    """Camera statistics"""
    camera_id: str
    running: bool
    frame_count: int
    fps: float
    queue_size: int
    frames_processed: int
    detections: int
    tracks_active: int
    persons_identified: int


class PersonInfo(BaseModel):
    """Person information"""
    person_id: int
    first_seen: str
    last_seen: str
    cameras_seen: List[str]
    total_appearances: int
    feature_count: int


class SystemStats(BaseModel):
    """System-wide statistics"""
    total_cameras: int
    cameras_running: int
    total_persons: int
    total_appearances: int
    cameras: dict


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str = "1.0.0"
    cameras: List[str]
    uptime_seconds: float
