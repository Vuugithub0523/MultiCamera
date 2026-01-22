"""
RTSP Stream Loader with Zero-Latency Buffering
Adapted from MultiCamera project for real-time streaming
"""
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class RTSPStreamLoader:
    """Thread-based RTSP stream loader with minimal latency"""
    
    def __init__(
        self,
        url: str,
        name: str,
        reconnect_delay: float = 2.0,
        buffer_size: int = 1,
        target_width: int = 640,
        target_height: int = 360,
    ) -> None:
        """
        Initialize RTSP stream loader
        
        Args:
            url: RTSP URL or video file path
            name: Stream name/identifier
            reconnect_delay: Seconds to wait before reconnection attempt
            buffer_size: OpenCV buffer size (1 = minimum latency)
            target_width: Target frame width for resize
            target_height: Target frame height for resize
        """
        self.url = url
        self.name = name
        self.reconnect_delay = reconnect_delay
        self.buffer_size = buffer_size
        self.target_width = target_width
        self.target_height = target_height
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[Tuple[float, np.ndarray]] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._update, daemon=True)
        
        # Statistics
        self.frame_count = 0
        self.fps = 0
        self._last_fps_time = time.time()
        self._fps_counter = 0

    def start(self) -> "RTSPStreamLoader":
        """Start the stream reading thread"""
        self._thread.start()
        print(f"[RTSPStreamLoader:{self.name}] Started reading from {self.url}")
        return self

    def stop(self) -> None:
        """Stop the stream reading thread"""
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        print(f"[RTSPStreamLoader:{self.name}] Stopped")

    def read(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Read the latest frame
        
        Returns:
            Tuple of (frame, timestamp) or (None, None) if no frame available
        """
        with self._lock:
            if self._frame is None:
                return None, None
            timestamp, frame = self._frame
        return frame.copy(), timestamp

    def is_opened(self) -> bool:
        """Check if stream is opened and reading"""
        with self._lock:
            return self._cap is not None and self._cap.isOpened()

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Open video capture with RTSP stream"""
        try:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            
            if not cap.isOpened():
                cap.release()
                return None
            
            print(f"[RTSPStreamLoader:{self.name}] Stream opened successfully")
            return cap
        except Exception as e:
            print(f"[RTSPStreamLoader:{self.name}] Error opening stream: {e}")
            return None

    def _update(self) -> None:
        """Background thread to continuously read frames"""
        reconnect_count = 0
        
        while not self._stop_event.is_set():
            # Open capture if not already opened
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._open_capture()
                
                if self._cap is None:
                    reconnect_count += 1
                    if reconnect_count % 5 == 0:
                        print(f"[RTSPStreamLoader:{self.name}] Reconnection attempt {reconnect_count}...")
                    time.sleep(self.reconnect_delay)
                    continue
                
                reconnect_count = 0

            # Read frame
            ok, frame = self._cap.read()
            
            if not ok:
                print(f"[RTSPStreamLoader:{self.name}] Failed to read frame, reconnecting...")
                self._cap.release()
                self._cap = None
                time.sleep(self.reconnect_delay)
                continue

            # Resize frame to target resolution for faster processing
            if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                frame = cv2.resize(
                    frame,
                    (self.target_width, self.target_height),
                    interpolation=cv2.INTER_LINEAR
                )

            # Update frame with timestamp
            timestamp = time.time()
            with self._lock:
                self._frame = (timestamp, frame)
            
            self.frame_count += 1
            self._update_fps()

    def _update_fps(self) -> None:
        """Update FPS counter"""
        self._fps_counter += 1
        now = time.time()
        elapsed = now - self._last_fps_time
        
        if elapsed >= 1.0:
            self.fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._last_fps_time = now


class MultiRTSPLoader:
    """Manager for multiple RTSP streams"""
    
    def __init__(self):
        self.loaders = {}
    
    def add_stream(self, camera_id: str, rtsp_url: str) -> RTSPStreamLoader:
        """
        Add and start a new RTSP stream
        
        Args:
            camera_id: Unique camera identifier
            rtsp_url: RTSP URL or video file path
        
        Returns:
            RTSPStreamLoader instance
        """
        if camera_id in self.loaders:
            print(f"[MultiRTSPLoader] Stream {camera_id} already exists")
            return self.loaders[camera_id]
        
        loader = RTSPStreamLoader(rtsp_url, camera_id).start()
        self.loaders[camera_id] = loader
        return loader
    
    def read_all(self) -> dict:
        """
        Read latest frames from all streams
        
        Returns:
            Dict mapping camera_id to (frame, timestamp)
        """
        frames = {}
        for camera_id, loader in self.loaders.items():
            frame, timestamp = loader.read()
            if frame is not None:
                frames[camera_id] = (frame, timestamp)
        return frames
    
    def stop_all(self) -> None:
        """Stop all streams"""
        for loader in self.loaders.values():
            loader.stop()
        self.loaders.clear()
        print("[MultiRTSPLoader] All streams stopped")
    
    def get_stats(self) -> dict:
        """Get statistics for all streams"""
        stats = {}
        for camera_id, loader in self.loaders.items():
            stats[camera_id] = {
                'is_opened': loader.is_opened(),
                'frame_count': loader.frame_count,
                'fps': round(loader.fps, 1)
            }
        return stats
