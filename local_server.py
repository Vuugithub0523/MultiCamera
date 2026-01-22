"""Local FastAPI server to stream RTSP cameras to the frontend via WebSocket."""

import asyncio
import signal
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import cv2
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from rtsp_multicam_loader import RTSPStreamLoader

DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_CAMERA_IDS = ("cam01", "cam02", "cam03")


class StreamManager:
    def __init__(self, rtsp_urls: List[str], camera_ids: List[str]) -> None:
        self.rtsp_urls = rtsp_urls
        self.camera_ids = camera_ids
        self.loaders: Dict[str, RTSPStreamLoader] = {}

    def start(self) -> None:
        for camera_id, url in zip(self.camera_ids, self.rtsp_urls):
            self.loaders[camera_id] = RTSPStreamLoader(url, camera_id).start()

    def stop(self) -> None:
        for loader in self.loaders.values():
            loader.stop()
        self.loaders.clear()

    def get_loader(self, camera_id: str) -> Optional[RTSPStreamLoader]:
        return self.loaders.get(camera_id)


def load_rtsp_urls(config_path: str) -> List[str]:
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    urls = payload.get("rtsp_urls")
    if not isinstance(urls, list) or not urls:
        raise ValueError("rtsp_urls must be a non-empty list in config.yaml")
    return [str(url) for url in urls]


@asynccontextmanager
async def lifespan(app: FastAPI):
    rtsp_urls = load_rtsp_urls(DEFAULT_CONFIG_PATH)
    camera_ids = list(DEFAULT_CAMERA_IDS[: len(rtsp_urls)])
    manager = StreamManager(rtsp_urls, camera_ids)
    manager.start()
    app.state.stream_manager = manager
    try:
        yield
    finally:
        manager.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws/{stream_type}/{camera_id}")
async def stream_camera(websocket: WebSocket, stream_type: str, camera_id: str) -> None:
    await websocket.accept()
    manager: StreamManager = websocket.app.state.stream_manager
    loader = manager.get_loader(camera_id)
    if loader is None:
        await websocket.close(code=1008)
        return

    try:
        while True:
            frame, _ = loader.read()
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                await asyncio.sleep(0.01)
                continue
            await websocket.send_bytes(buffer.tobytes())
            await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        return


def main() -> None:
    import uvicorn

    uvicorn.run("local_server:app", host="0.0.0.0", port=8080, reload=False)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: None)
    main()
