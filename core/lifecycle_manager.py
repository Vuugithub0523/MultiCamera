"""
Person Lifecycle Manager
Manages the lifecycle of tracked persons across cameras
Adapted from MultiCamera project
"""
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from collections import defaultdict
from typing import Optional, Dict, List, Tuple


class PersonState(Enum):
    """Person lifecycle states"""
    DETECTED = "detected"           # First detection
    TRACKING = "tracking"           # Being tracked
    LOST = "lost"                   # Temporarily lost
    CONFIRMED_LOST = "confirmed_lost"  # Confirmed gone
    ARCHIVED = "archived"           # Archived


class PersonLifecycle:
    """Manages lifecycle of a single person"""
    
    def __init__(self, person_id: int, camera_id: str, confidence: float, bbox: tuple):
        self.person_id = person_id
        self.state = PersonState.DETECTED
        
        # Basic info
        self.first_seen = datetime.now()
        self.last_seen = datetime.now()
        self.last_camera = camera_id
        self.current_camera = camera_id
        
        # History
        self.detections_history = []
        self.camera_history = [camera_id]
        self.state_history = [(PersonState.DETECTED, datetime.now())]
        
        # Statistics
        self.total_detections = 1
        self.cameras_visited = {camera_id: 1}
        self.confidences = [confidence]
        
        # Frame tracking
        self.frames_missing = 0
        self.max_frames_missing = 0
        
        # Add first detection
        self._add_detection(camera_id, confidence, bbox)
    
    def _add_detection(self, camera_id: str, confidence: float, bbox: tuple, match_info: Optional[Dict] = None):
        """Add a detection to history"""
        detection = {
            'timestamp': datetime.now().isoformat(),
            'camera_id': camera_id,
            'confidence': confidence,
            'bbox': bbox,
            'state': self.state.value
        }
        
        if match_info:
            detection.update({
                'match_score': match_info.get('match_score'),
                'matched_global_id': match_info.get('matched_global_id'),
                'match_confidence': match_info.get('match_confidence'),
                'reasoning': match_info.get('reasoning'),
                'feasibility_reason': match_info.get('feasibility_reason')
            })
        
        self.detections_history.append(detection)
    
    def update(self, camera_id: str, confidence: float, bbox: tuple, match_info: Optional[Dict] = None):
        """Update when person is detected"""
        now = datetime.now()
        
        # Reset frames missing
        if self.frames_missing > 0:
            self.max_frames_missing = max(self.max_frames_missing, self.frames_missing)
            self.frames_missing = 0
        
        # Update state to TRACKING
        if self.state != PersonState.TRACKING:
            self._change_state(PersonState.TRACKING)
        
        # Update timestamps
        self.last_seen = now
        self.last_camera = self.current_camera
        self.current_camera = camera_id
        
        # Update camera history
        if not self.camera_history or self.camera_history[-1] != camera_id:
            self.camera_history.append(camera_id)
        
        # Update statistics
        self.total_detections += 1
        self.cameras_visited[camera_id] = self.cameras_visited.get(camera_id, 0) + 1
        self.confidences.append(confidence)
        
        # Add detection to history
        self._add_detection(camera_id, confidence, bbox, match_info)
    
    def mark_missing(self):
        """Mark frame as missing (not detected)"""
        self.frames_missing += 1
    
    def mark_lost(self):
        """Mark as temporarily lost"""
        if self.state == PersonState.TRACKING:
            self._change_state(PersonState.LOST)
    
    def mark_confirmed_lost(self):
        """Mark as confirmed lost"""
        if self.state == PersonState.LOST:
            self._change_state(PersonState.CONFIRMED_LOST)
    
    def archive(self):
        """Archive the person"""
        if self.state == PersonState.CONFIRMED_LOST:
            self._change_state(PersonState.ARCHIVED)
    
    def _change_state(self, new_state: PersonState):
        """Change lifecycle state"""
        old_state = self.state
        self.state = new_state
        self.state_history.append((new_state, datetime.now()))
        
        # Log state transition
        print(f"[Lifecycle] Person {self.person_id}: {old_state.value} -> {new_state.value}")
    
    def get_summary(self) -> Dict:
        """Get summary of person's lifecycle"""
        duration = (self.last_seen - self.first_seen).total_seconds()
        
        return {
            'person_id': self.person_id,
            'state': self.state.value,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'duration_seconds': duration,
            'current_camera': self.current_camera,
            'cameras_visited': list(self.cameras_visited.keys()),
            'total_detections': self.total_detections,
            'avg_confidence': sum(self.confidences) / len(self.confidences),
            'max_frames_missing': self.max_frames_missing
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'person_id': self.person_id,
            'state': self.state.value,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'current_camera': self.current_camera,
            'camera_history': self.camera_history,
            'total_detections': self.total_detections,
            'cameras_visited': dict(self.cameras_visited),
            'detections_history': self.detections_history[-10:]  # Last 10 detections
        }


class PersonLifecycleManager:
    """Manager for all person lifecycles"""
    
    def __init__(self, output_dir: str = "./tracking_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Active persons being tracked
        self.persons: Dict[int, PersonLifecycle] = {}
        
        # Archived persons
        self.archived_persons: Dict[int, PersonLifecycle] = {}
        
        # Configuration
        self.max_lost_frames = 30           # LOST threshold
        self.max_confirm_lost_frames = 90   # CONFIRMED_LOST threshold
        self.archive_after_seconds = 300    # ARCHIVED threshold
        
        # ID counter
        self.next_person_id = 1
        
        # Statistics
        self.total_persons_created = 0
        self.total_persons_archived = 0
        self.time_window_rejections = 0
        
        print(f"[LifecycleManager] Initialized with output_dir: {output_dir}")
    
    def create_person(
        self, 
        camera_id: str, 
        confidence: float, 
        bbox: tuple,
        match_info: Optional[Dict] = None
    ) -> int:
        """Create new person"""
        person_id = self.next_person_id
        self.next_person_id += 1
        
        person = PersonLifecycle(person_id, camera_id, confidence, bbox)
        
        # Add match info if provided
        if match_info:
            person._add_detection(camera_id, confidence, bbox, match_info)
        
        self.persons[person_id] = person
        self.total_persons_created += 1
        
        print(f"[LifecycleManager] Created person {person_id} at camera {camera_id}")
        return person_id
    
    def update_person(
        self, 
        person_id: int, 
        camera_id: str, 
        confidence: float, 
        bbox: tuple,
        match_info: Optional[Dict] = None
    ):
        """Update existing person"""
        if person_id in self.persons:
            self.persons[person_id].update(camera_id, confidence, bbox, match_info)
        else:
            print(f"[LifecycleManager] Warning: Person {person_id} not found")
    
    def mark_frame_end(self, detected_ids: List[int]):
        """Mark end of frame - update missing counts"""
        for person_id, person in list(self.persons.items()):
            if person_id not in detected_ids:
                person.mark_missing()
                
                # Check transitions
                if person.state == PersonState.TRACKING and person.frames_missing >= self.max_lost_frames:
                    person.mark_lost()
                
                elif person.state == PersonState.LOST and person.frames_missing >= self.max_confirm_lost_frames:
                    person.mark_confirmed_lost()
    
    def cleanup_old_persons(self):
        """Archive old confirmed lost persons"""
        now = datetime.now()
        to_archive = []
        
        for person_id, person in self.persons.items():
            if person.state == PersonState.CONFIRMED_LOST:
                time_since_last_seen = (now - person.last_seen).total_seconds()
                
                if time_since_last_seen >= self.archive_after_seconds:
                    to_archive.append(person_id)
        
        # Archive persons
        for person_id in to_archive:
            self._archive_person(person_id)
    
    def _archive_person(self, person_id: int):
        """Archive a person"""
        if person_id not in self.persons:
            return
        
        person = self.persons[person_id]
        person.archive()
        
        # Move to archived
        self.archived_persons[person_id] = person
        del self.persons[person_id]
        
        self.total_persons_archived += 1
        
        # Save to file
        self._save_person_log(person)
        
        print(f"[LifecycleManager] Archived person {person_id}")
    
    def _save_person_log(self, person: PersonLifecycle):
        """Save person log to JSON file"""
        filename = self.output_dir / f"person_{person.person_id}_{person.first_seen.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(person.to_dict(), f, indent=2)
    
    def get_matchable_persons(self, current_time: datetime, time_window_seconds: float) -> List[int]:
        """Get persons within time window for matching"""
        matchable = []
        threshold = current_time - timedelta(seconds=time_window_seconds)
        
        for person_id, person in self.persons.items():
            if person.last_seen >= threshold:
                matchable.append(person_id)
        
        return matchable
    
    def get_active_persons(self) -> Dict[int, PersonLifecycle]:
        """Get all active (non-archived) persons"""
        return self.persons.copy()
    
    def get_person(self, person_id: int) -> Optional[PersonLifecycle]:
        """Get specific person by ID"""
        return self.persons.get(person_id)
    
    def get_stats(self) -> Dict:
        """Get lifecycle statistics"""
        state_counts = defaultdict(int)
        for person in self.persons.values():
            state_counts[person.state.value] += 1
        
        return {
            'total_active': len(self.persons),
            'total_archived': len(self.archived_persons),
            'total_created': self.total_persons_created,
            'state_distribution': dict(state_counts),
            'time_window_rejections': self.time_window_rejections
        }
    
    def export_summary_csv(self, filename: str = "tracking_summary.csv"):
        """Export summary of all persons to CSV"""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'person_id', 'state', 'first_seen', 'last_seen', 'duration_seconds',
                'current_camera', 'cameras_visited', 'total_detections', 
                'avg_confidence', 'max_frames_missing'
            ])
            
            writer.writeheader()
            
            # Write active persons
            for person in self.persons.values():
                writer.writerow(person.get_summary())
            
            # Write archived persons
            for person in self.archived_persons.values():
                writer.writerow(person.get_summary())
        
        print(f"[LifecycleManager] Exported summary to {filepath}")
