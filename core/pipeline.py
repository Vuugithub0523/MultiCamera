"""
Camera Pipeline
Processes frames from camera: detection -> tracking -> re-id -> output
Enhanced with lifecycle management and better visualization
"""
import cv2
import numpy as np
import time
from datetime import datetime
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

from detection import YOLODetector, BYTETracker
from reid import FeatureExtractor, PersonDatabase
from core.lifecycle_manager import PersonLifecycleManager


@dataclass
class TrackInfo:
    """Information about a tracked object"""
    track_id: int
    person_id: Optional[int]
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    is_new: bool = False
    state: Optional[str] = None  # Person state from lifecycle


class CameraPipeline:
    """Processing pipeline for a single camera with lifecycle management"""
    
    def __init__(
        self,
        camera_id: str,
        detector: YOLODetector,
        tracker: BYTETracker,
        feature_extractor: FeatureExtractor,
        person_db: PersonDatabase,
        lifecycle_manager: PersonLifecycleManager,
        detect_skip_frames: int = 2,
        output_fps: int = 15,
        reid_threshold: float = 0.42,
        time_window_seconds: float = 3.0,
        camera_topology: Optional[Dict[str, List[str]]] = None,
        camera_transition_max_time: Optional[Dict[str, float]] = None
    ):
        """
        Initialize camera pipeline
        
        Args:
            camera_id: Camera identifier
            detector: YOLO detector (shared)
            tracker: BYTETracker instance (per-camera)
            feature_extractor: Feature extractor (shared)
            person_db: Person database (shared)
            lifecycle_manager: Lifecycle manager (shared)
            detect_skip_frames: Run detection every N frames
            output_fps: Output frame rate for WebSocket
            reid_threshold: Re-ID matching threshold
            time_window_seconds: Time window for person matching
            camera_topology: Camera connection topology dict
            camera_transition_max_time: Max transition times between cameras
        """
        self.camera_id = camera_id
        self.detector = detector
        self.tracker = tracker
        self.feature_extractor = feature_extractor
        self.person_db = person_db
        self.lifecycle_manager = lifecycle_manager
        self.detect_skip_frames = detect_skip_frames
        self.output_fps = output_fps
        self.reid_threshold = reid_threshold
        self.time_window_seconds = time_window_seconds
        self.camera_topology = camera_topology or {}
        self.camera_transition_max_time = camera_transition_max_time or {}
        
        self.frame_count = 0
        self.track_to_person = {}  # Map track_id to person_id
        self.person_colors = {}    # Map person_id to color for consistent visualization
        self.last_output_time = 0
        self.output_interval = 1.0 / output_fps if output_fps > 0 else 0
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'detections': 0,
            'tracks_active': 0,
            'persons_identified': 0,
            'fps': 0,
            'last_fps_time': time.time(),
            'fps_counter': 0
        }
    
    async def process_frame(self, frame: np.ndarray) -> Tuple[bytes, List[TrackInfo]]:
        """
        Process a single frame through the pipeline
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            Tuple of (JPEG bytes, list of track info)
        """
        self.frame_count += 1
        self.stats['frames_processed'] += 1
        current_time = datetime.now()
        detected_person_ids = []
        
        # Step 1: Detection (skip frames for performance)
        if self.frame_count % self.detect_skip_frames == 0:
            detections = self.detector.detect(frame)
            self.stats['detections'] += len(detections)
            print(f"[{self.camera_id}] Frame {self.frame_count}: Detected {len(detections)} objects")
        else:
            detections = np.empty((0, 5))
        
        # Step 2: Tracking (every frame)
        tracks = self.tracker.update(detections, img_info=frame.shape[:2])
        self.stats['tracks_active'] = len(tracks)
        print(f"[{self.camera_id}] Frame {self.frame_count}: Active tracks: {len(tracks)}")
        
        # Step 3: Feature extraction and Re-ID with lifecycle management
        track_infos = []
        for track in tracks:
            x, y, w, h = map(int, track.tlwh)
            
            # Check if this track already has a person ID
            person_id = self.track_to_person.get(track.track_id)
            is_new = person_id is None
            person_state = None
            
            # Extract features for new tracks
            if is_new and track.is_activated:
                # Crop person region
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    features = self.feature_extractor.extract(crop)
                    
                    if features is not None:
                        # Get matchable persons using topology-based matching
                        if self.camera_topology:
                            # Use topology-based matching
                            matchable_persons_dict = self.lifecycle_manager.get_matchable_persons_topology(
                                self.camera_id,
                                current_time,
                                self.time_window_seconds,
                                self.camera_topology,
                                self.camera_transition_max_time
                            )
                            # Extract just the person IDs
                            matchable_persons = list(matchable_persons_dict.keys())
                            
                            # Log topology decision
                            if matchable_persons_dict:
                                reasons = [reason for _, reason in matchable_persons_dict.values()]
                                print(f"[{self.camera_id}] Topology matching: {len(matchable_persons)} candidates - {reasons[:3]}")
                        else:
                            # Fallback to time window only
                            matchable_persons = self.lifecycle_manager.get_matchable_persons(
                                current_time,
                                self.time_window_seconds
                            )
                        
                        # Try to match with existing persons
                        if matchable_persons:
                            # Try matching with person database
                            person_id = self.person_db.find_match(
                                features,
                                threshold=self.reid_threshold,
                                camera_id=self.camera_id
                            )
                            
                            if person_id and person_id in matchable_persons:
                                # Matched with existing person
                                self.person_db.update_person(person_id, features, self.camera_id)
                                
                                # Update lifecycle
                                match_info = {
                                    'match_score': None,
                                    'matched_global_id': person_id,
                                    'match_confidence': track.score,
                                    'reasoning': 'reid_matched',
                                    'feasibility_reason': 'within_time_window'
                                }
                                self.lifecycle_manager.update_person(
                                    person_id,
                                    self.camera_id,
                                    track.score,
                                    (x, y, w, h),
                                    match_info
                                )
                            else:
                                # Create new person
                                person_id = self.lifecycle_manager.create_person(
                                    camera_id=self.camera_id,
                                    confidence=track.score,
                                    bbox=(x, y, w, h),
                                    match_info={
                                        'reasoning': 'no_match_in_window',
                                        'feasibility_reason': 'new_person'
                                    }
                                )
                                self.person_db.add_person(features, self.camera_id)
                                self.stats['persons_identified'] += 1
                                
                                # Assign random color for visualization
                                if person_id not in self.person_colors:
                                    self.person_colors[person_id] = tuple(
                                        int(c) for c in np.random.randint(0, 255, size=3)
                                    )
                        else:
                            # No matchable persons - create new
                            person_id = self.lifecycle_manager.create_person(
                                camera_id=self.camera_id,
                                confidence=track.score,
                                bbox=(x, y, w, h)
                            )
                            self.person_db.add_person(features, self.camera_id)
                            self.stats['persons_identified'] += 1
                            
                            # Assign random color
                            if person_id not in self.person_colors:
                                self.person_colors[person_id] = tuple(
                                    int(c) for c in np.random.randint(0, 255, size=3)
                                )
                        
                        self.track_to_person[track.track_id] = person_id
            
            # Get person state from lifecycle
            if person_id:
                person = self.lifecycle_manager.get_person(person_id)
                if person:
                    person_state = person.state.value
                    detected_person_ids.append(person_id)
            
            # Create track info
            track_info = TrackInfo(
                track_id=track.track_id,
                person_id=person_id,
                bbox=(x, y, w, h),
                confidence=track.score,
                is_new=is_new,
                state=person_state
            )
            track_infos.append(track_info)
        
        # Step 4: Update lifecycle - mark missing persons
        self.lifecycle_manager.mark_frame_end(detected_person_ids)
        self.lifecycle_manager.cleanup_old_persons()
        
        # Step 5: Encode original frame to JPEG (no annotation - frontend will draw)
        # This saves CPU/RAM by skipping frame copy and drawing operations
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpeg_bytes = jpeg.tobytes()
        
        # Update FPS
        self._update_fps()
        
        return jpeg_bytes, track_infos
    
    def _draw_annotations(self, frame: np.ndarray, tracks: List[TrackInfo]) -> np.ndarray:
        """Draw bounding boxes and IDs on frame with enhanced visualization"""
        annotated = frame.copy()
        
        for track in tracks:
            x, y, w, h = track.bbox
            
            # Color based on person state and ID
            if track.person_id:
                # Use consistent color for each person
                if track.person_id in self.person_colors:
                    color = self.person_colors[track.person_id]
                else:
                    color = (0, 255, 0)  # Default green
                
                # Modify color based on state
                if track.state == "lost":
                    color = tuple(c // 2 for c in color)  # Dim color for lost
                elif track.state == "confirmed_lost":
                    color = (128, 128, 128)  # Gray
            else:
                color = (0, 255, 255)  # Yellow for unknown tracks
            
            # Draw bounding box
            thickness = 3 if track.person_id else 2
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
            
            # Prepare label
            if track.person_id:
                label = f"ID:{track.person_id}"
                if track.state:
                    state_short = {
                        'detected': 'DET',
                        'tracking': 'TRK',
                        'lost': 'LST',
                        'confirmed_lost': 'CLT'
                    }.get(track.state, track.state[:3].upper())
                    label += f" [{state_short}]"
            else:
                label = f"T{track.track_id}"
            
            # Add confidence
            label += f" {track.confidence:.2f}"
            
            # Draw label with background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Position label above bbox
            label_y = max(y - 10, label_h + 10)
            label_x = x
            
            # Draw label background
            cv2.rectangle(
                annotated,
                (label_x, label_y - label_h - baseline),
                (label_x + label_w, label_y + baseline),
                color,
                -1
            )
            
            # Draw label text (black for visibility)
            cv2.putText(
                annotated,
                label,
                (label_x, label_y - baseline),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness
            )
        
        # Draw camera info and statistics
        info_lines = [
            f"Camera: {self.camera_id}",
            f"FPS: {self.stats['fps']:.1f}",
            f"Tracks: {len(tracks)}",
            f"Persons: {self.stats['persons_identified']}"
        ]
        
        y_offset = 25
        for line in info_lines:
            cv2.putText(
                annotated,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            y_offset += 25
        
        # Draw lifecycle stats
        lifecycle_stats = self.lifecycle_manager.get_stats()
        lifecycle_text = f"Active: {lifecycle_stats['total_active']} | Archived: {lifecycle_stats['total_archived']}"
        cv2.putText(
            annotated,
            lifecycle_text,
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1
        )
        
        return annotated
    
    def _update_fps(self):
        """Update FPS statistics"""
        self.stats['fps_counter'] += 1
        now = time.time()
        elapsed = now - self.stats['last_fps_time']
        
        if elapsed >= 1.0:
            self.stats['fps'] = self.stats['fps_counter'] / elapsed
            self.stats['fps_counter'] = 0
            self.stats['last_fps_time'] = now
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        return {
            'camera_id': self.camera_id,
            **self.stats
        }
    
    def cleanup(self):
        """Cleanup pipeline resources"""
        self.track_to_person.clear()
