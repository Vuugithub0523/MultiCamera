"""
Person Database for Re-Identification
Stores and manages person features for cross-camera tracking
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class Person:
    """Person data structure"""
    person_id: int
    first_seen: str
    last_seen: str
    cameras_seen: List[str]
    total_appearances: int
    feature_gallery: List[List[float]]  # List of feature vectors
    
    def to_dict(self):
        """Convert to dictionary"""
        return asdict(self)
    
    @staticmethod
    def from_dict(data):
        """Create from dictionary"""
        return Person(**data)


class PersonDatabase:
    """Database for storing and matching person features"""
    
    def __init__(self, db_file: str, max_gallery_size: int = 512):
        """
        Initialize person database
        
        Args:
            db_file: Path to JSON database file
            max_gallery_size: Maximum features to store per person
        """
        self.db_file = Path(db_file)
        self.max_gallery_size = max_gallery_size
        self.persons: Dict[int, Person] = {}
        self.next_id = 1
        
        # Create storage directory if needed
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing database
        self.load()
    
    def load(self):
        """Load database from file"""
        if self.db_file.exists():
            try:
                with open(self.db_file, 'r') as f:
                    data = json.load(f)
                    self.persons = {
                        int(pid): Person.from_dict(pdata)
                        for pid, pdata in data.get('persons', {}).items()
                    }
                    self.next_id = data.get('next_id', 1)
                print(f"[PersonDB] Loaded {len(self.persons)} persons from {self.db_file}")
            except Exception as e:
                print(f"[PersonDB] Error loading database: {e}")
                self.persons = {}
                self.next_id = 1
        else:
            print(f"[PersonDB] No existing database found, starting fresh")
    
    def save(self):
        """Save database to file"""
        try:
            data = {
                'persons': {
                    str(pid): person.to_dict()
                    for pid, person in self.persons.items()
                },
                'next_id': self.next_id
            }
            with open(self.db_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[PersonDB] Error saving database: {e}")
    
    def add_person(self, features: np.ndarray, camera_id: str) -> int:
        """
        Add new person to database
        
        Args:
            features: Feature vector
            camera_id: Camera where person was first seen
        
        Returns:
            Person ID
        """
        person_id = self.next_id
        self.next_id += 1
        
        now = datetime.now().isoformat()
        person = Person(
            person_id=person_id,
            first_seen=now,
            last_seen=now,
            cameras_seen=[camera_id],
            total_appearances=1,
            feature_gallery=[features.tolist()]
        )
        
        self.persons[person_id] = person
        self.save()
        
        print(f"[PersonDB] Added new person {person_id} from camera {camera_id}")
        return person_id
    
    def update_person(
        self,
        person_id: int,
        features: np.ndarray,
        camera_id: str
    ):
        """
        Update existing person with new observation
        
        Args:
            person_id: Person ID to update
            features: New feature vector
            camera_id: Camera where person was seen
        """
        if person_id not in self.persons:
            print(f"[PersonDB] Warning: Person {person_id} not found")
            return
        
        person = self.persons[person_id]
        person.last_seen = datetime.now().isoformat()
        person.total_appearances += 1
        
        if camera_id not in person.cameras_seen:
            person.cameras_seen.append(camera_id)
        
        # Add to gallery (keep only recent ones)
        person.feature_gallery.append(features.tolist())
        if len(person.feature_gallery) > self.max_gallery_size:
            person.feature_gallery = person.feature_gallery[-self.max_gallery_size:]
        
        # Save periodically (every 10 appearances)
        if person.total_appearances % 10 == 0:
            self.save()
    
    def find_match(
        self,
        features: np.ndarray,
        threshold: float = 0.42,
        camera_id: Optional[str] = None
    ) -> Optional[int]:
        """
        Find matching person in database
        
        Args:
            features: Query feature vector
            threshold: Distance threshold for match
            camera_id: Current camera (optional, for logging)
        
        Returns:
            Person ID if match found, None otherwise
        """
        if len(self.persons) == 0:
            return None
        
        best_match = None
        best_distance = float('inf')
        
        for person_id, person in self.persons.items():
            # Compute distance to all gallery features
            gallery = np.array(person.feature_gallery)
            distances = np.array([
                self._cosine_distance(features, gfeat)
                for gfeat in gallery
            ])
            
            # Use minimum distance
            min_dist = np.min(distances)
            
            if min_dist < best_distance:
                best_distance = min_dist
                best_match = person_id
        
        if best_distance < threshold:
            print(f"[PersonDB] Matched person {best_match} (distance: {best_distance:.3f})")
            return best_match
        
        return None
    
    @staticmethod
    def _cosine_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine distance between two feature vectors"""
        feat1 = np.array(feat1).flatten()
        feat2 = np.array(feat2).flatten()
        
        dot = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0
        
        similarity = dot / (norm1 * norm2)
        distance = 1.0 - similarity
        
        return max(0.0, min(1.0, distance))
    
    def get_person(self, person_id: int) -> Optional[Person]:
        """Get person by ID"""
        return self.persons.get(person_id)
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            'total_persons': len(self.persons),
            'next_id': self.next_id,
            'total_appearances': sum(p.total_appearances for p in self.persons.values())
        }
