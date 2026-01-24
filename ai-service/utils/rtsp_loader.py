"""Multi-camera RTSP loader with zero-latency buffering.

Quick start (Linux/macOS):
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install opencv-python numpy
  python rtsp_multicam_loader.py

Notes:
  - Update RTSP URLs in main() if your credentials or IPs change.
  - For GPU-accelerated decode, install an OpenCV build with FFmpeg + HW decode.
  - Press "q" in the display window to quit.
"""

import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class RTSPStreamLoader:
    def __init__(
        self,
        url: str,
        name: str,
        reconnect_delay: float = 2.0,
        buffer_size: int = 1,
    ) -> None:
        self.url = url
        self.name = name
        self.reconnect_delay = reconnect_delay
        self.buffer_size = buffer_size
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[Tuple[float, "cv2.Mat"]] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._update, daemon=True)

    def start(self) -> "RTSPStreamLoader":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()

    def read(self) -> Tuple[Optional["cv2.Mat"], Optional[float]]:
        with self._lock:
            if self._frame is None:
                return None, None
            timestamp, frame = self._frame
        return frame, timestamp

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _update(self) -> None:
        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._open_capture()
                if self._cap is None:
                    time.sleep(self.reconnect_delay)
                    continue

            ok, frame = self._cap.read()
            if not ok:
                self._cap.release()
                self._cap = None
                time.sleep(self.reconnect_delay)
                continue

            timestamp = time.time()
            with self._lock:
                self._frame = (timestamp, frame)
