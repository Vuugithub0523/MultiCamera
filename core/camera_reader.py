"""
RTSP Camera Reader
Reads frames from RTSP stream with minimal latency
"""
import cv2
import numpy as np
import threading
import time
from queue import Queue, Empty
from typing import Optional, Tuple


class RTSPReader:
    """RTSP stream reader with threaded frame capture"""
    
    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        buffer_size: int = 2,
        target_width: int = 1280,
        target_height: int = 720
    ):
        """
        Initialize RTSP reader
        
        Args:
            camera_id: Camera identifier
            rtsp_url: RTSP URL or video file path
            buffer_size: Frame buffer size (smaller = lower latency)
            target_width: Target frame width
            target_height: Target frame height
        """
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.buffer_size = buffer_size
        self.target_width = target_width
        self.target_height = target_height
        
        self.frame_queue = Queue(maxsize=buffer_size)
        self.running = False
        self.thread = None
        self.capture = None
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
    
    def start(self):
        """Start reading frames in background thread"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        
        print(f"[RTSPReader:{self.camera_id}] Started reading from {self.rtsp_url}")
    
    def stop(self):
        """Stop reading frames"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.capture:
            self.capture.release()
        
        print(f"[RTSPReader:{self.camera_id}] Stopped")
    
    def _read_loop(self):
        """Background thread to read frames"""
        retry_count = 0
        max_retries = 5
        
        while self.running and retry_count < max_retries:
            try:
                # Open capture
                self.capture = cv2.VideoCapture(self.rtsp_url)
                
                # Set buffer size to minimum for low latency
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Set resolution if possible
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                
                if not self.capture.isOpened():
                    print(f"[RTSPReader:{self.camera_id}] Failed to open stream, retrying...")
                    retry_count += 1
                    time.sleep(2.0)
                    continue
                
                print(f"[RTSPReader:{self.camera_id}] Stream opened successfully")
                retry_count = 0  # Reset on success
                
                # Read frames
                while self.running:
                    ret, frame = self.capture.read()
                    
                    if not ret:
                        print(f"[RTSPReader:{self.camera_id}] Failed to read frame, reconnecting...")
                        break
                    
                    # Resize to target size if needed
                    if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                        frame = cv2.resize(
                            frame,
                            (self.target_width, self.target_height),
                            interpolation=cv2.INTER_LINEAR
                        )
                    
                    # Drop old frames if queue is full (maintain low latency)
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            pass
                    
                    self.frame_queue.put(frame)
                    self.frame_count += 1
                    
                    # Update FPS counter
                    self._update_fps()
                
                # Clean up capture before retry
                if self.capture:
                    self.capture.release()
                    self.capture = None
                
                time.sleep(1.0)
                
            except Exception as e:
                print(f"[RTSPReader:{self.camera_id}] Error: {e}")
                retry_count += 1
                time.sleep(2.0)
        
        if retry_count >= max_retries:
            print(f"[RTSPReader:{self.camera_id}] Max retries exceeded, giving up")
    
    def _update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        now = time.time()
        elapsed = now - self.last_fps_time
        
        if elapsed >= 1.0:
            self.fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.last_fps_time = now
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get latest frame
        
        Args:
            timeout: Timeout in seconds
        
        Returns:
            Frame as numpy array or None if timeout
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def is_alive(self) -> bool:
        """Check if reader thread is alive"""
        return self.running and (self.thread is not None and self.thread.is_alive())
    
    def get_stats(self) -> dict:
        """Get reader statistics"""
        return {
            'camera_id': self.camera_id,
            'running': self.running,
            'frame_count': self.frame_count,
            'fps': self.fps,
            'queue_size': self.frame_queue.qsize()
        }
