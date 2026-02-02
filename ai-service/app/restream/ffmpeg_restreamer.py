"""
FFmpeg Restreamer
Re-stream annotated frames to RTSP via FFmpeg subprocess.
"""

import asyncio
import logging
import subprocess
import time
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FFmpegRestreamer:
    """
    Re-streams frames to RTSP using FFmpeg subprocess.
    Input: rawvideo from stdin
    Output: H.264 RTSP stream to MediaMTX
    """
    
    def __init__(
        self,
        output_url: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 15,
        preset: str = "veryfast",
        tune: str = "zerolatency",
        bitrate: str = "2M",
    ):
        self.output_url = output_url
        self.width = width
        self.height = height
        self.fps = fps
        self.preset = preset
        self.tune = tune
        self.bitrate = bitrate
        
        self._process: Optional[subprocess.Popen] = None
        self._running = False
        self._lock = asyncio.Lock()
        self._frames_written = 0
        self._start_time = 0.0
        self._restart_count = 0
    
    async def start(self):
        """Start the FFmpeg process."""
        async with self._lock:
            if self._running:
                return
            
            self._start_process()
            self._running = True
            self._start_time = time.time()
    
    async def stop(self):
        """Stop the FFmpeg process."""
        async with self._lock:
            self._running = False
            self._stop_process()
    
    def _start_process(self):
        """Start FFmpeg subprocess."""
        import os
        ffmpeg_bin = os.environ.get("FFMPEG_PATH") or "ffmpeg"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",  # stdin
            "-c:v", "libx264",
            "-preset", self.preset,
            "-tune", self.tune,
            "-b:v", self.bitrate,
            "-maxrate", self.bitrate,
            "-bufsize", "1M",
            "-pix_fmt", "yuv420p",
            "-g", str(self.fps * 2),  # GOP size
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            self.output_url,
        ]
        
        logger.info(f"Starting FFmpeg: {' '.join(cmd)}")
        
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # Drain stderr to avoid blocking and to surface errors
        try:
            def _drain_stderr(proc: subprocess.Popen, url: str):
                assert proc.stderr is not None
                for raw in iter(proc.stderr.readline, b""):
                    line = raw.decode(errors="ignore").strip()
                    if line:
                        logger.debug(f"FFmpeg[{url}]: {line}")

            import threading
            threading.Thread(
                target=_drain_stderr,
                args=(self._process, self.output_url),
                daemon=True,
            ).start()
        except Exception as e:
            logger.warning(f"Failed to attach FFmpeg stderr reader: {e}")

        logger.info(f"FFmpeg started for {self.output_url}")
    
    def _stop_process(self):
        """Stop FFmpeg subprocess."""
        if self._process:
            try:
                self._process.stdin.close()
                self._process.wait(timeout=5.0)
            except:
                self._process.kill()
            self._process = None
    
    async def write_frame(self, frame: np.ndarray):
        """
        Write a frame to FFmpeg stdin.
        Frame must be BGR format with correct dimensions.
        """
        if not self._running or self._process is None:
            return
        
        # Ensure correct dimensions
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            import cv2
            frame = cv2.resize(frame, (self.width, self.height))
        
        try:
            # Write frame bytes to stdin
            self._process.stdin.write(frame.tobytes())
            self._process.stdin.flush()
            self._frames_written += 1
            
        except (BrokenPipeError, OSError) as e:
            logger.warning(f"FFmpeg pipe error: {e}, restarting...")
            await self._restart()
    
    async def _restart(self):
        """Restart FFmpeg process."""
        async with self._lock:
            self._stop_process()
            await asyncio.sleep(0.5)
            self._start_process()
            self._restart_count += 1
            logger.info(f"FFmpeg restarted for {self.output_url}")
    
    def get_stats(self) -> Dict:
        """Get restreamer statistics."""
        elapsed = time.time() - self._start_time if self._start_time > 0 else 0
        fps_out = self._frames_written / elapsed if elapsed > 0 else 0
        
        return {
            "output_url": self.output_url,
            "frames_written": self._frames_written,
            "fps_out": round(fps_out, 2),
            "restart_count": self._restart_count,
            "running": self._running and self._process is not None,
        }
