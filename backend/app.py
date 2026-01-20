from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from rtsp_multicam_loader import RTSPStreamLoader


@dataclass(frozen=True)
class CameraConfig:
    id: str
    code: str
    name: str
    location: str
    rtsp_url: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_camera_configs() -> List[CameraConfig]:
    config_path = _repo_root() / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yaml at {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    rtsp_urls = config.get("rtsp_urls", [])
    cameras = []
    for index, url in enumerate(rtsp_urls, start=1):
        cam_id = f"cam{index:02d}"
        cameras.append(
            CameraConfig(
                id=cam_id,
                code=f"CCTV {index:02d}",
                name=f"Camera {index}",
                location=f"Zone {index}",
                rtsp_url=url,
            )
        )
    return cameras


class StreamRegistry:
    def __init__(self, cameras: List[CameraConfig]) -> None:
        self._cameras = {camera.id: camera for camera in cameras}
        self._loaders: Dict[str, RTSPStreamLoader] = {}

    def start(self) -> None:
        for camera in self._cameras.values():
            if camera.id not in self._loaders:
                self._loaders[camera.id] = RTSPStreamLoader(camera.rtsp_url, camera.id).start()

    def stop(self) -> None:
        for loader in self._loaders.values():
            loader.stop()
        self._loaders.clear()

    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        loader = self._loaders.get(camera_id)
        if loader is None:
            return None
        frame, _ = loader.read()
        return frame

    def get_camera(self, camera_id: str) -> Optional[CameraConfig]:
        return self._cameras.get(camera_id)

    def list_cameras(self) -> List[CameraConfig]:
        return list(self._cameras.values())


def _build_placeholder(camera_id: str, message: str) -> np.ndarray:
    width, height = 1280, 720
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        f"{camera_id}",
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (0, 255, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        message,
        (40, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def _encode_jpeg(frame: np.ndarray) -> Optional[bytes]:
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return buffer.tobytes()


registry = StreamRegistry(_load_camera_configs())


@asynccontextmanager
async def lifespan(_: FastAPI):
    registry.start()
    try:
        yield
    finally:
        registry.stop()


app = FastAPI(title="MultiCamera Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.get("/cameras")
async def list_cameras() -> JSONResponse:
    cameras = registry.list_cameras()
    data = [
        {
            "id": camera.id,
            "code": camera.code,
            "name": camera.name,
            "location": camera.location,
            "rtspUrl": camera.rtsp_url,
            "streamUrl": f"http://localhost:8080/api/stream/{camera.id}",
        }
        for camera in cameras
    ]
    return JSONResponse({"ok": True, "data": data})


@app.get("/api/stream/{camera_id}")
async def stream_camera(camera_id: str) -> StreamingResponse:
    camera = registry.get_camera(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    async def frame_generator():
        while True:
            frame = registry.get_frame(camera_id)
            if frame is None:
                frame = _build_placeholder(camera_id, "Waiting for stream...")
            jpg = _encode_jpeg(frame)
            if jpg is None:
                await asyncio.sleep(0.05)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )
            await asyncio.sleep(0.03)

    return StreamingResponse(
        frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws/stream/{camera_id}")
async def stream_camera_ws(websocket: WebSocket, camera_id: str) -> None:
    camera = registry.get_camera(camera_id)
    if camera is None:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    try:
        while True:
            frame = registry.get_frame(camera_id)
            if frame is None:
                frame = _build_placeholder(camera_id, "Connecting to camera...")
            jpg = _encode_jpeg(frame)
            if jpg is None:
                await asyncio.sleep(0.05)
                continue
            await websocket.send_bytes(jpg)
            await asyncio.sleep(0.03)
    except WebSocketDisconnect:
        return


@app.websocket("/ws/tracking/{camera_id}")
async def stream_tracking_ws(websocket: WebSocket, camera_id: str) -> None:
    await stream_camera_ws(websocket, camera_id)
