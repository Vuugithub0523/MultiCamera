import threading
import time
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml
from scipy.spatial import distance

# Add AI service to path
sys.path.insert(0, str(Path(__file__).parent.parent / "ai-service"))

from core.object_detection import ObjectDetection
from core.feature_extraction import FeatureExtraction
from core.person_lifecycle_manager import PersonLifecycleManager, PersonState
from utils.rtsp_loader import RTSPStreamLoader

DEFAULT_CAMERA_IDS = ("cam01", "cam02", "cam03")


@dataclass
class TrackingConfig:
    """Configuration for tracking pipeline"""
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
    """Manages RTSP streams and tracking pipeline"""
    
    def __init__(
        self,
        rtsp_urls: List[str],
        camera_ids: List[str],
        tracking_config: TrackingConfig,
    ) -> None:
        self.rtsp_urls = rtsp_urls
        self.camera_ids = camera_ids
        self.tracking_config = tracking_config
        
        # Stream loaders
        self.loaders: Dict[str, RTSPStreamLoader] = {}
        
        # Frame storage
        self._raw_frames: Dict[str, "cv2.Mat"] = {}
        self._tracking_frames: Dict[str, "cv2.Mat"] = {}
        self._lock = threading.Lock()
        
        # Processing thread
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        
        # Tracking pipeline
        self._tracking_pipeline: Optional[TrackingPipeline] = None
        
        # Camera ID mappings
        self._camera_index_by_id = {camera_id: idx for idx, camera_id in enumerate(self.camera_ids)}
        self._camera_id_by_index = {idx: camera_id for idx, camera_id in enumerate(self.camera_ids)}

    def start(self) -> None:
        """Start stream loaders and processing thread"""
        # Start RTSP loaders
        for camera_id, url in zip(self.camera_ids, self.rtsp_urls):
            self.loaders[camera_id] = RTSPStreamLoader(url, camera_id).start()
        
        # Initialize tracking pipeline if enabled
        if self.tracking_config.enabled:
            self._tracking_pipeline = TrackingPipeline(self.tracking_config)
        
        # Start processing thread
        self._thread.start()

    def stop(self) -> None:
        """Stop all streams and processing"""
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        
        # Stop all loaders
        for loader in self.loaders.values():
            loader.stop()
        self.loaders.clear()
        
        # Save tracking results
        if self._tracking_pipeline:
            self._tracking_pipeline.lifecycle_manager.save_summary()
            self._tracking_pipeline.lifecycle_manager.print_final_report()
        
        self._tracking_pipeline = None
        
        # Clear frames
        with self._lock:
            self._raw_frames.clear()
            self._tracking_frames.clear()

    def get_loader(self, camera_id: str) -> Optional[RTSPStreamLoader]:
        """Get RTSP loader for a camera"""
        return self.loaders.get(camera_id)

    def get_latest_frame(self, camera_id: str, stream_type: str) -> Optional["cv2.Mat"]:
        """Get latest frame for a camera"""
        with self._lock:
            if stream_type == "tracking":
                return self._tracking_frames.get(camera_id)
            return self._raw_frames.get(camera_id)

    def _run_loop(self) -> None:
        """Main processing loop - runs in separate thread"""
        interval = max(1, self.tracking_config.interval)
        frame_index = 0
        
        while not self._stop_event.is_set():
            # Read frames from all cameras
            frames_by_id: Dict[str, "cv2.Mat"] = {}
            for camera_id, loader in self.loaders.items():
                frame, _ = loader.read()
                if frame is None:
                    continue
                frames_by_id[camera_id] = frame
            
            if not frames_by_id:
                time.sleep(0.01)
                continue

            # Update raw frames
            with self._lock:
                self._raw_frames.update(frames_by_id)

            # Run tracking pipeline at intervals
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


class TrackingPipeline:
    """Multi-camera person tracking pipeline"""
    
    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        
        # Initialize AI models
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
        
        # Initialize lifecycle manager
        self.lifecycle_manager = PersonLifecycleManager(output_dir=config.tracking_log_dir)
        self.lifecycle_manager.max_lost_frames = config.max_lost_frames
        self.lifecycle_manager.max_confirm_lost_frames = config.max_confirm_lost_frames
        self.lifecycle_manager.archive_after_seconds = config.archive_after_seconds
        
        # Person database
        self.detected_persons: Dict[str, Dict[str, object]] = {}

    def _draw_detection(
        self,
        frame: "cv2.Mat",
        person_id: int,
        cls_name: str,
        confidence: float,
        bbox: tuple[int, int, int, int],
    ) -> None:
        """Draw bounding box and label on frame"""
        person = self.lifecycle_manager.active_persons.get(person_id)
        
        if person is None:
            color = (128, 128, 128)
            state_label = "unkn"
        elif person.state == PersonState.DETECTED:
            color = (0, 255, 0)
            state_label = person.state.value[:4]
        elif person.state == PersonState.TRACKING:
            color = self.detected_persons[f"id_{person_id}"]["color"].tolist()
            state_label = person.state.value[:4]
        else:
            color = (128, 128, 128)
            state_label = person.state.value[:4]

        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {person_id} [{state_label}]: {confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            color,
            2,
        )

    def _get_matchable_persons(self, current_time: datetime) -> Dict[int, object]:
        """Get persons that can be matched in current frame"""
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
        """Process frames from all cameras and perform tracking"""
        current_time = datetime.now()
        detected_ids_in_frame: List[int] = []
        predictions: Dict[str, List[Dict[str, Dict[str, object]]]] = {}
        
        # Create annotated copies
        annotated_frames = {camera_id: frame.copy() for camera_id, frame in frames_by_id.items()}

        # Run detection on all frames
        for camera_id, frame in frames_by_id.items():
            predictions[camera_id] = self.detector.predict_img(frame)

        # Process detections
        for camera_id, detections in predictions.items():
            camera_index = camera_index_by_id[camera_id]
            frame = frames_by_id[camera_id]
            
            for predict in detections:
                cls_name = tuple(predict.keys())[0]
                x1, y1, x2, y2 = predict[cls_name]["bounding_box"]
                
                # Validate bbox
                height, width = frame.shape[:2]
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Extract features from person crop
                cropped_image = frame[y1:y2, x1:x2]
                extracted_features = self.feature_extractor.predict_img(cropped_image)[0]

                # First person ever detected
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
                    self._draw_detection(
                        annotated_frames[camera_id],
                        person_id,
                        cls_name,
                        predict[cls_name]["confidence"],
                        (x1, y1, x2, y2),
                    )
                    continue

                # Get matchable persons (within time window)
                matchable_persons = self._get_matchable_persons(current_time)
                
                if not matchable_persons:
                    # No matchable persons - create new
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
                    self._draw_detection(
                        annotated_frames[camera_id],
                        person_id,
                        cls_name,
                        predict[cls_name]["confidence"],
                        (x1, y1, x2, y2),
                    )
                    continue

                # Find best match among candidates
                candidates = []
                for person_id, person in matchable_persons.items():
                    key = f"id_{person_id}"
                    if key not in self.detected_persons:
                        continue
                    
                    value = self.detected_persons[key]
                    
                    # Calculate cosine distance
                    score = distance.cosine(
                        np.mean(value["extracted_features"], axis=0)
                        if len(value["extracted_features"]) > 1
                        else value["extracted_features"].flatten(),
                        extracted_features.flatten(),
                    )
                    
                    candidates.append({
                        "id": value["id"],
                        "cls_name": value["cls_name"],
                        "color": value["color"],
                        "score": score,
                        "time_diff": person.get_time_since_last_seen(current_time),
                    })

                if candidates:
                    # Get best match
                    top1_person = sorted(candidates, key=lambda d: d["score"])[0]
                    
                    if top1_person["score"] < self.config.feature_threshold:
                        # Match found - update existing person
                        self.lifecycle_manager.update_person(
                            person_id=top1_person["id"],
                            camera_id=camera_index,
                            confidence=predict[cls_name]["confidence"],
                            bbox=(x1, y1, x2, y2),
                        )
                        
                        # Update feature gallery
                        key = f"id_{top1_person['id']}"
                        existing_features = self.detected_persons[key]["extracted_features"]
                        
                        if existing_features.shape[0] < self.config.max_gallery_set_each_person:
                            updated_features = np.vstack((existing_features, extracted_features))
                        else:
                            # FIFO - remove oldest feature
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
                        self._draw_detection(
                            annotated_frames[camera_id],
                            top1_person["id"],
                            top1_person["cls_name"],
                            predict[cls_name]["confidence"],
                            (x1, y1, x2, y2),
                        )
                    else:
                        # No good match - create new person
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
                        self._draw_detection(
                            annotated_frames[camera_id],
                            person_id,
                            cls_name,
                            predict[cls_name]["confidence"],
                            (x1, y1, x2, y2),
                        )
                else:
                    # No candidates - create new person
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
                    self._draw_detection(
                        annotated_frames[camera_id],
                        person_id,
                        cls_name,
                        predict[cls_name]["confidence"],
                        (x1, y1, x2, y2),
                    )

        # Update lifecycle states
        self.lifecycle_manager.process_frame_end(detected_ids_in_frame)
        
        return annotated_frames


def load_rtsp_urls(config_path: str) -> List[str]:
    """Load RTSP URLs from config file"""
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    
    urls = payload.get("rtsp_urls")
    if not isinstance(urls, list) or not urls:
        raise ValueError("rtsp_urls must be a non-empty list in config.yaml")
    
    return [str(url) for url in urls]


def load_tracking_config(config_path: str) -> TrackingConfig:
    """Load tracking configuration from config file"""
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
    """Build camera IDs based on number of RTSP URLs"""
    camera_ids = list(DEFAULT_CAMERA_IDS)
    
    if len(rtsp_urls) <= len(camera_ids):
        return camera_ids[: len(rtsp_urls)]
    
    for idx in range(len(camera_ids), len(rtsp_urls)):
        camera_ids.append(f"cam{idx + 1:02d}")
    
    return camera_ids
