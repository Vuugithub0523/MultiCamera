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

from helpers import resize_with_letterbox

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


def process_tracking(
    frames: Dict[str, "cv2.Mat"],
    frame_index: int,
    detect_interval: int = 3,
) -> None:
    if frame_index % detect_interval == 0:
        for name, frame in frames.items():
            _ = (name, frame)
            # TODO: Run detector (YOLO) on this frame.
    else:
        for name, frame in frames.items():
            _ = (name, frame)
            # TODO: Run tracker (Kalman Filter/SORT) update on this frame.


def _validate_resolution(
    frames: Dict[str, "cv2.Mat"],
    min_resolution: int,
    warned: Dict[str, bool],
) -> None:
    if min_resolution <= 0:
        return
    for name, frame in frames.items():
        if warned.get(name, False):
            continue
        height, width = frame.shape[:2]
        if min(height, width) < min_resolution:
            print(
                f"[WARN] {name} resolution {width}x{height} < {min_resolution}; "
                "accuracy may degrade for detection."
            )
            warned[name] = True


def _build_mosaic(
    frames: Dict[str, "cv2.Mat"],
    order: Tuple[str, ...],
    tile_size: Tuple[int, int],
    scale: float,
) -> "cv2.Mat":
    tile_w, tile_h = tile_size
    tiles = []
    for name in order:
        frame = frames.get(name)
        if frame is None:
            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        else:
            tile, _, _ = resize_with_letterbox(frame, (tile_w, tile_h))
        tiles.append(tile)

    mosaic = np.hstack(tiles)
    if scale != 1.0:
        mosaic = cv2.resize(
            mosaic, (int(mosaic.shape[1] * scale), int(mosaic.shape[0] * scale))
        )
    return mosaic


def main() -> None:
    urls = {
        "cam1": "rtsp://admin:1234567Chuong@192.168.1.254:554/cam/realmonitor?channel=1&subtype=1",
        "cam2": "rtsp://admin:123456Chuong@192.168.1.254:554/cam/realmonitor?channel=1&subtype=1",
        "cam3": "rtsp://admin:12345678Chuong@192.168.1.254:554/cam/realmonitor?channel=1&subtype=1",
    }
    display_enabled = True
    display_tile_size = (1020, 920)
    display_scale = 0.5
    min_detection_resolution = 640  # Set to 0 to disable resolution warnings.

    loaders = {name: RTSPStreamLoader(url, name).start() for name, url in urls.items()}
    frame_index = 0
    fps_start = time.time()
    fps_counter = 0
    warned: Dict[str, bool] = {}

    try:
        while True:
            frames: Dict[str, "cv2.Mat"] = {}
            for name, loader in loaders.items():
                frame, _ = loader.read()
                if frame is None:
                    continue
                frames[name] = frame

            if len(frames) != len(loaders):
                time.sleep(0.005)
                continue

            _validate_resolution(frames, min_detection_resolution, warned)
            process_tracking(frames, frame_index)
            frame_index += 1

            if display_enabled:
                mosaic = _build_mosaic(
                    frames,
                    order=("cam1", "cam2", "cam3"),
                    tile_size=display_tile_size,
                    scale=display_scale,
                )
                cv2.imshow("MultiCamera", mosaic)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                print(f"[INFO] Main loop FPS: {fps:.2f}")
                fps_counter = 0
                fps_start = time.time()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if display_enabled:
            cv2.destroyAllWindows()
        for loader in loaders.values():
            loader.stop()


if __name__ == "__main__":
    main()
