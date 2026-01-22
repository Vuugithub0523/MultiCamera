import asyncio
import signal
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from scipy.spatial import distance

from feature_extraction import FeatureExtraction
from object_detection import ObjectDetection
from person_lifecycle_manager import PersonLifecycleManager, PersonState
from rtsp_multicam_loader import RTSPStreamLoader

DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_CAMERA_IDS = ("cam01", "cam02", "cam03")


@dataclass
class TrackingConfig:
    enabled: bool
    interval: int
    confidence_threshold: float
    model_path: str
    classes_path: str
    device: str
    feature_model_path: str
    feature_threshold: float
    max_gallery_set_each_person: int
    time_window_seconds: float
    camera_topology: Dict[int, List[int]]
    camera_transition_max_time: Dict[str, float]
    max_lost_frames: int
    max_confirm_lost_frames: int
    archive_after_seconds: int
    tracking_log_dir: str


class StreamManager:
    def __init__(
        self,
        rtsp_urls: List[str],
        camera_ids: List[str],
        tracking_config: TrackingConfig,
    ) -> None:
        self.rtsp_urls = rtsp_urls
        self.camera_ids = camera_ids
        self.tracking_config = tracking_config
        self.loaders: Dict[str, RTSPStreamLoader] = {}
        self._raw_frames: Dict[str, "cv2.Mat"] = {}
        self._tracking_frames: Dict[str, "cv2.Mat"] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._tracking_pipeline: Optional[TrackingPipeline] = None
        self._camera_index_by_id = {camera_id: idx for idx, camera_id in enumerate(self.camera_ids)}
        self._camera_id_by_index = {idx: camera_id for idx, camera_id in enumerate(self.camera_ids)}

    def start(self) -> None:
        for camera_id, url in zip(self.camera_ids, self.rtsp_urls):
            self.loaders[camera_id] = RTSPStreamLoader(url, camera_id).start()
        if self.tracking_config.enabled:
            self._tracking_pipeline = TrackingPipeline(self.tracking_config)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        for loader in self.loaders.values():
            loader.stop()
        self.loaders.clear()
        self._tracking_pipeline = None
        with self._lock:
            self._raw_frames.clear()
            self._tracking_frames.clear()

    def get_loader(self, camera_id: str) -> Optional[RTSPStreamLoader]:
        return self.loaders.get(camera_id)

    def get_latest_frame(self, camera_id: str, stream_type: str) -> Optional["cv2.Mat"]:
        with self._lock:
            if stream_type == "tracking":
                return self._tracking_frames.get(camera_id)
            return self._raw_frames.get(camera_id)

    def _run_loop(self) -> None:
        interval = max(1, self.tracking_config.interval)
        frame_index = 0
        while not self._stop_event.is_set():
            frames_by_id: Dict[str, "cv2.Mat"] = {}
            for camera_id, loader in self.loaders.items():
                frame, _ = loader.read()
                if frame is None:
                    continue
                frames_by_id[camera_id] = frame
            if not frames_by_id:
                time.sleep(0.01)
                continue

            with self._lock:
                self._raw_frames.update(frames_by_id)

            if self._tracking_pipeline and frame_index % interval == 0:
                annotated = self._tracking_pipeline.process(
                    frames_by_id,
                    self._camera_index_by_id,
                    self._camera_id_by_index,
                )
                with self._lock:
                    self._tracking_frames.update(annotated)
            frame_index += 1
            time.sleep(0.005)


def load_rtsp_urls(config_path: str) -> List[str]:
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    urls = payload.get("rtsp_urls")
    if not isinstance(urls, list) or not urls:
        raise ValueError("rtsp_urls must be a non-empty list in config.yaml")
    return [str(url) for url in urls]


def load_tracking_config(config_path: str) -> TrackingConfig:
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return TrackingConfig(
        enabled=bool(payload.get("tracking_stream_enabled", True)),
        interval=int(payload.get("tracking_stream_interval", 3)),
        confidence_threshold=float(payload.get("object_detection_threshold", 0.7)),
        model_path=str(payload.get("object_detection_model_path", "")),
        classes_path=str(payload.get("object_detection_classes_path", "")),
        device=str(payload.get("inference_model_device", "cpu")),
        feature_model_path=str(payload.get("feature_extraction_model_path", "")),
        feature_threshold=float(payload.get("feature_extraction_threshold", 0.42)),
        max_gallery_set_each_person=int(payload.get("max_gallery_set_each_person", 512)),
        time_window_seconds=float(payload.get("time_window_seconds", 3.0)),
        camera_topology=payload.get("camera_topology", {}) or {},
        camera_transition_max_time=payload.get("camera_transition_max_time", {}) or {},
        max_lost_frames=int(payload.get("max_lost_frames", 30)),
        max_confirm_lost_frames=int(payload.get("max_confirm_lost_frames", 90)),
        archive_after_seconds=int(payload.get("archive_after_seconds", 300)),
        tracking_log_dir=str(payload.get("tracking_log_dir", "./tracking_logs")),
    )


def build_camera_ids(rtsp_urls: List[str]) -> List[str]:
    camera_ids = list(DEFAULT_CAMERA_IDS)
    if len(rtsp_urls) <= len(camera_ids):
        return camera_ids[: len(rtsp_urls)]
    for idx in range(len(camera_ids), len(rtsp_urls)):
        camera_ids.append(f"cam{idx + 1:02d}")
    return camera_ids


class TrackingPipeline:
    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        self.detector = ObjectDetection(
            confidence_threshold=config.confidence_threshold,
            onnx_path=config.model_path,
            coco_names_path=config.classes_path,
            device=config.device,
        )
        self.feature_extractor = FeatureExtraction(
            onnx_path=config.feature_model_path,
            device=config.device,
        )
        self.lifecycle_manager = PersonLifecycleManager(output_dir=config.tracking_log_dir)
        self.lifecycle_manager.max_lost_frames = config.max_lost_frames
        self.lifecycle_manager.max_confirm_lost_frames = config.max_confirm_lost_frames
        self.lifecycle_manager.archive_after_seconds = config.archive_after_seconds
        self.detected_persons: Dict[str, Dict[str, object]] = {}

    def _get_matchable_persons(
        self,
        current_time: datetime,
        ) -> Dict[int, object]:
        return self.lifecycle_manager.get_matchable_persons(
            current_time,
            self.config.time_window_seconds,
        )

    def process(
        self,
        frames_by_id: Dict[str, "cv2.Mat"],
        camera_index_by_id: Dict[str, int],
        camera_id_by_index: Dict[int, str],
    ) -> Dict[str, "cv2.Mat"]:
        current_time = datetime.now()
        detected_ids_in_frame: List[int] = []
        predictions: Dict[str, List[Dict[str, Dict[str, object]]]] = {}
        annotated_frames = {camera_id: frame.copy() for camera_id, frame in frames_by_id.items()}

        for camera_id, frame in frames_by_id.items():
            predictions[camera_id] = self.detector.predict_img(frame)

        for camera_id, detections in predictions.items():
            camera_index = camera_index_by_id[camera_id]
            frame = frames_by_id[camera_id]
            for predict in detections:
                cls_name = tuple(predict.keys())[0]
                x1, y1, x2, y2 = predict[cls_name]["bounding_box"]
                height, width = frame.shape[:2]
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                cropped_image = frame[y1:y2, x1:x2]
                extracted_features = self.feature_extractor.predict_img(cropped_image)[0]

                if not self.detected_persons:
                    person_id = self.lifecycle_manager.create_person(
                        camera_id=camera_index,
                        confidence=predict[cls_name]["confidence"],
                        bbox=(x1, y1, x2, y2),
                    )
                    self.detected_persons[f"id_{person_id}"] = {
                        "extracted_features": extracted_features,
                        "id": person_id,
                        "camera_id": camera_index,
                        "cls_name": cls_name,
                        "bbox": (x1, y1, x2, y2),
                        "confidence": predict[cls_name]["confidence"],
                        "color": np.random.randint(0, 255, size=3),
                    }
                    detected_ids_in_frame.append(person_id)
                    continue

                matchable_persons = self._get_matchable_persons(current_time)
                if not matchable_persons:
                    if self.detected_persons:
                        self.lifecycle_manager.time_window_rejections += 1
                    person_id = self.lifecycle_manager.create_person(
                        camera_id=camera_index,
                        confidence=predict[cls_name]["confidence"],
                        bbox=(x1, y1, x2, y2),
                    )
                    self.detected_persons[f"id_{person_id}"] = {
                        "extracted_features": extracted_features,
                        "id": person_id,
                        "camera_id": camera_index,
                        "cls_name": cls_name,
                        "bbox": (x1, y1, x2, y2),
                        "confidence": predict[cls_name]["confidence"],
                        "color": np.random.randint(0, 255, size=3),
                    }
                    detected_ids_in_frame.append(person_id)
                    continue

                candidates = []
                for person_id, person in matchable_persons.items():
                    key = f"id_{person_id}"
                    if key not in self.detected_persons:
                        continue
                    value = self.detected_persons[key]
                    score = distance.cosine(
                        np.mean(value["extracted_features"], axis=0)
                        if len(value["extracted_features"]) > 1
                        else value["extracted_features"].flatten(),
                        extracted_features.flatten(),
                    )
                    candidates.append(
                        {
                            "id": value["id"],
                            "cls_name": value["cls_name"],
                            "color": value["color"],
                            "score": score,
                            "time_diff": person.get_time_since_last_seen(current_time),
                        }
                    )

                if candidates:
                    top1_person = sorted(candidates, key=lambda d: d["score"])[0]
                    if top1_person["score"] < self.config.feature_threshold:
                        self.lifecycle_manager.update_person(
                            person_id=top1_person["id"],
                            camera_id=camera_index,
                            confidence=predict[cls_name]["confidence"],
                            bbox=(x1, y1, x2, y2),
                        )
                        key = f"id_{top1_person['id']}"
                        existing_features = self.detected_persons[key]["extracted_features"]
                        if existing_features.shape[0] < self.config.max_gallery_set_each_person:
                            updated_features = np.vstack((existing_features, extracted_features))
                        else:
                            updated_features = np.vstack((extracted_features, existing_features[1:]))
                        self.detected_persons[key] = {
                            "extracted_features": updated_features,
                            "id": top1_person["id"],
                            "camera_id": camera_index,
                            "cls_name": top1_person["cls_name"],
                            "bbox": (x1, y1, x2, y2),
                            "confidence": predict[cls_name]["confidence"],
                            "color": top1_person["color"],
                        }
                        detected_ids_in_frame.append(top1_person["id"])
                    else:
                        person_id = self.lifecycle_manager.create_person(
                            camera_id=camera_index,
                            confidence=predict[cls_name]["confidence"],
                            bbox=(x1, y1, x2, y2),
                        )
                        self.detected_persons[f"id_{person_id}"] = {
                            "extracted_features": extracted_features,
                            "id": person_id,
                            "camera_id": camera_index,
                            "cls_name": cls_name,
                            "bbox": (x1, y1, x2, y2),
                            "confidence": predict[cls_name]["confidence"],
                            "color": np.random.randint(0, 255, size=3),
                        }
                        detected_ids_in_frame.append(person_id)
                else:
                    person_id = self.lifecycle_manager.create_person(
                        camera_id=camera_index,
                        confidence=predict[cls_name]["confidence"],
                        bbox=(x1, y1, x2, y2),
                    )
                    self.detected_persons[f"id_{person_id}"] = {
                        "extracted_features": extracted_features,
                        "id": person_id,
                        "camera_id": camera_index,
                        "cls_name": cls_name,
                        "bbox": (x1, y1, x2, y2),
                        "confidence": predict[cls_name]["confidence"],
                        "color": np.random.randint(0, 255, size=3),
                    }
                    detected_ids_in_frame.append(person_id)

        active_persons = self.lifecycle_manager.active_persons
        for person_id, person in active_persons.items():
            if person.current_camera not in camera_id_by_index:
                continue
            camera_id = camera_id_by_index[person.current_camera]
            if camera_id not in annotated_frames:
                continue
            key = f"id_{person_id}"
            if key not in self.detected_persons:
                continue
            value = self.detected_persons[key]
            if person.state == PersonState.DETECTED:
                color = (0, 255, 0)
            elif person.state == PersonState.TRACKING:
                color = value["color"].tolist()
            else:
                color = (128, 128, 128)
            x1, y1, x2, y2 = value["bbox"]
            cv2.rectangle(annotated_frames[camera_id], (x1, y1), (x2, y2), color, 2)
            label = f"{value['cls_name']} {person_id} [{person.state.value[:4]}]: {value['confidence']:.2f}"
            cv2.putText(
                annotated_frames[camera_id],
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                color,
                2,
            )

        self.lifecycle_manager.process_frame_end(detected_ids_in_frame)
        return annotated_frames


@asynccontextmanager
async def lifespan(app: FastAPI):
    rtsp_urls = load_rtsp_urls(DEFAULT_CONFIG_PATH)
    camera_ids = build_camera_ids(rtsp_urls)
    tracking_config = load_tracking_config(DEFAULT_CONFIG_PATH)
    manager = StreamManager(rtsp_urls, camera_ids, tracking_config)
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
    if manager.get_loader(camera_id) is None:
        await websocket.close(code=1008)
        return

    try:
        while True:
            frame = manager.get_latest_frame(camera_id, stream_type)
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