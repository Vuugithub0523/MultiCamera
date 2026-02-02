"""
RTSP Ingest Module
Low-latency RTSP stream ingestion with frame dropping and auto-reconnect.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import threading
import queue

import cv2
import numpy as np

from app.config import CameraConfig

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Container for frame data with metadata."""
    frame: np.ndarray
    timestamp: float
    fps: float


class RTSPIngestWorker:
    """
    Worker thread for RTSP stream ingestion.
    Keeps only the latest frame (drops old frames for low latency).
    """
    
    def __init__(self, cam_id: str, rtsp_url: str):
        self.cam_id = cam_id
        self.rtsp_url = rtsp_url
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[FrameData] = None
        self._frame_lock = threading.Lock()
        self._reconnect_count = 0
        self._fps_counter = 0
        self._fps_time = time.time()
        self._current_fps = 0.0
        self._connected = False
        self._stats_lock = threading.Lock()
    
    def start(self):
        """Start the ingest worker thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Started RTSP ingest worker for {self.cam_id}")
    
    def stop(self):
        """Stop the ingest worker thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info(f"Stopped RTSP ingest worker for {self.cam_id}")
    
    def _run(self):
        """Main worker loop."""
        while self._running:
            try:
                self._connect_and_read()
            except Exception as e:
                logger.error(f"RTSP ingest error for {self.cam_id}: {e}")
                with self._stats_lock:
                    self._connected = False
                    self._reconnect_count += 1
                time.sleep(2.0)  # Wait before reconnect
    
    def _connect_and_read(self):
        """Connect to RTSP stream and read frames."""
        logger.info(f"Connecting to RTSP: {self.cam_id}")
        
        # OpenCV VideoCapture with optimized settings
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        # Set buffer size to 1 for low latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Try to set other optimizations
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        
        if not cap.isOpened():
            logger.error(f"Failed to open RTSP stream: {self.cam_id}")
            raise ConnectionError(f"Cannot open RTSP: {self.rtsp_url}")
        
        with self._stats_lock:
            self._connected = True
        logger.info(f"Connected to RTSP: {self.cam_id}")
        
        try:
            while self._running:
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    logger.warning(f"Lost frame from {self.cam_id}, reconnecting...")
                    break
                
                # Update FPS counter
                self._fps_counter += 1
                now = time.time()
                elapsed = now - self._fps_time
                if elapsed >= 1.0:
                    with self._stats_lock:
                        self._current_fps = self._fps_counter / elapsed
                    self._fps_counter = 0
                    self._fps_time = now
                
                # Keep only latest frame
                with self._frame_lock:
                    self._latest_frame = FrameData(
                        frame=frame,
                        timestamp=now,
                        fps=self._current_fps,
                    )
        finally:
            cap.release()
            with self._stats_lock:
                self._connected = False
    
    def get_frame(self) -> Optional[Tuple[np.ndarray, float, float]]:
        """
        Get the latest frame.
        Returns: (frame, timestamp, fps) or None if no frame available.
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            data = self._latest_frame
            self._latest_frame = None  # Clear after reading
            return (data.frame, data.timestamp, data.fps)
    
    def get_status(self) -> str:
        """Get connection status."""
        with self._stats_lock:
            return "connected" if self._connected else "disconnected"
    
    def get_stats(self) -> Dict:
        """Get statistics."""
        with self._stats_lock:
            return {
                "connected": self._connected,
                "fps_ingest": round(self._current_fps, 2),
                "reconnect_count": self._reconnect_count,
            }


class RTSPIngestManager:
    """
    Manager for multiple RTSP ingest workers.
    """
    
    def __init__(self, cameras: List[CameraConfig]):
        self.cameras = cameras
        self._workers: Dict[str, RTSPIngestWorker] = {}
        
        for cam in cameras:
            self._workers[cam.id] = RTSPIngestWorker(cam.id, cam.rtsp)
    
    async def start(self):
        """Start all ingest workers."""
        for worker in self._workers.values():
            worker.start()
            await asyncio.sleep(0.1)  # Stagger starts
    
    async def stop(self):
        """Stop all ingest workers."""
        for worker in self._workers.values():
            worker.stop()
    
    async def get_frame(self, cam_id: str) -> Optional[Tuple[np.ndarray, float, float]]:
        """Get latest frame for a camera."""
        if cam_id not in self._workers:
            return None
        return self._workers[cam_id].get_frame()
    
    async def get_status(self, cam_id: str) -> str:
        """Get status for a camera."""
        if cam_id not in self._workers:
            return "unknown"
        return self._workers[cam_id].get_status()
    
    async def get_stats(self, cam_id: str) -> Dict:
        """Get stats for a camera."""
        if cam_id not in self._workers:
            return {}
        return self._workers[cam_id].get_stats()
