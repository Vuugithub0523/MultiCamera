import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from collections import defaultdict


class PersonState(Enum):
    """Trạng thái vòng đời của một person"""
    DETECTED = "detected"           # Mới phát hiện lần đầu
    TRACKING = "tracking"           # Đang theo dõi
    LOST = "lost"                   # Tạm thời mất dấu
    CONFIRMED_LOST = "confirmed_lost"  # Xác nhận đã rời đi
    ARCHIVED = "archived"           # Đã lưu trữ


class PersonLifecycle:
    """Quản lý vòng đời của một person"""
    
    def __init__(self, person_id, camera_id, confidence, bbox):
        self.person_id = person_id
        self.state = PersonState.DETECTED
        
        # Thông tin cơ bản
        self.first_seen = datetime.now()
        self.last_seen = datetime.now()
        self.last_camera = camera_id
        self.current_camera = camera_id
        
        # Lịch sử tracking
        self.detections_history = []
        self.camera_history = [camera_id]
        self.state_history = [(PersonState.DETECTED, datetime.now())]
        
        # Thống kê
        self.total_detections = 1
        self.cameras_visited = {camera_id: 1}
        self.confidences = [confidence]
        
        # Frame tracking (để phát hiện lost)
        self.frames_missing = 0
        self.max_frames_missing = 0
        
        # Thêm detection đầu tiên
        self._add_detection(camera_id, confidence, bbox)
    
    def _add_detection(self, camera_id, confidence, bbox, match_info=None):
        """
        Thêm một detection vào lịch sử
        
        Args:
            camera_id: ID của camera
            confidence: Độ tin cậy từ detector
            bbox: Bounding box
            match_info: Dict chứa thông tin matching {
                'match_score': float,
                'matched_global_id': int or None,
                'match_confidence': float,
                'reasoning': str,
                'feasibility_reason': str or None
            }
        """
        detection = {
            'timestamp': datetime.now().isoformat(),
            'camera_id': camera_id,
            'confidence': confidence,
            'bbox': bbox,
            'state': self.state.value
        }
        
        # Thêm match metadata nếu có
        if match_info:
            detection.update({
                'match_score': match_info.get('match_score'),
                'matched_global_id': match_info.get('matched_global_id'),
                'match_confidence': match_info.get('match_confidence'),
                'reasoning': match_info.get('reasoning'),
                'feasibility_reason': match_info.get('feasibility_reason')
            })
        
        self.detections_history.append(detection)
    
    def update(self, camera_id, confidence, bbox, match_info=None):
        """
        Cập nhật khi phát hiện person
        Transition: DETECTED -> TRACKING hoặc LOST -> TRACKING
        
        Args:
            camera_id: ID của camera
            confidence: Độ tin cậy
            bbox: Bounding box
            match_info: Dict chứa metadata về matching
        """
        now = datetime.now()
        
        # Reset frames missing
        if self.frames_missing > 0:
            self.max_frames_missing = max(self.max_frames_missing, self.frames_missing)
            self.frames_missing = 0
        
        # Cập nhật state
        old_state = self.state
        if self.state == PersonState.DETECTED:
            self.state = PersonState.TRACKING
            self._add_state_change(old_state, PersonState.TRACKING)
        elif self.state == PersonState.LOST:
            self.state = PersonState.TRACKING
            self._add_state_change(old_state, PersonState.TRACKING)
            print(f"   🔄 Person {self.person_id}: LOST -> TRACKING (tìm lại sau {self.max_frames_missing} frames)")
        
        # Cập nhật thông tin
        self.last_seen = now
        self.last_camera = self.current_camera
        self.current_camera = camera_id
        self.total_detections += 1
        self.confidences.append(confidence)
        
        # Cập nhật camera history
        if camera_id not in self.camera_history or self.camera_history[-1] != camera_id:
            self.camera_history.append(camera_id)
        
        # Cập nhật camera visits
        if camera_id in self.cameras_visited:
            self.cameras_visited[camera_id] += 1
        else:
            self.cameras_visited[camera_id] = 1
        
        # Thêm detection với match info
        self._add_detection(camera_id, confidence, bbox, match_info)
    
    def mark_missing(self):
        """
        Đánh dấu person không được phát hiện trong frame hiện tại
        Transition: TRACKING -> LOST (sau N frames)
        """
        self.frames_missing += 1
        
        # Chuyển sang LOST sau 30 frames không thấy
        if self.frames_missing == 30 and self.state == PersonState.TRACKING:
            old_state = self.state
            self.state = PersonState.LOST
            self._add_state_change(old_state, PersonState.LOST)
            print(f"   ⚠️  Person {self.person_id}: TRACKING -> LOST (mất {self.frames_missing} frames)")
    
    def confirm_lost(self):
        """
        Xác nhận person đã rời đi hẳn
        Transition: LOST -> CONFIRMED_LOST
        """
        if self.state == PersonState.LOST:
            old_state = self.state
            self.state = PersonState.CONFIRMED_LOST
            self._add_state_change(old_state, PersonState.CONFIRMED_LOST)
            print(f"   ❌ Person {self.person_id}: LOST -> CONFIRMED_LOST (mất {self.frames_missing} frames)")
            return True
        return False
    
    def archive(self):
        """
        Lưu trữ person
        Transition: CONFIRMED_LOST -> ARCHIVED
        """
        if self.state == PersonState.CONFIRMED_LOST:
            old_state = self.state
            self.state = PersonState.ARCHIVED
            self._add_state_change(old_state, PersonState.ARCHIVED)
            print(f"   📦 Person {self.person_id}: CONFIRMED_LOST -> ARCHIVED")
            return True
        return False
    
    def _add_state_change(self, old_state, new_state):
        """Ghi lại sự thay đổi state"""
        self.state_history.append((new_state, datetime.now()))
    
    def get_duration(self):
        """Tính tổng thời gian (giây)"""
        return (self.last_seen - self.first_seen).total_seconds()
    
    def is_within_time_window(self, current_time, time_window_seconds):
        """
        Kiểm tra xem person có nằm trong time window không
        Args:
            current_time: datetime object của thời điểm hiện tại
            time_window_seconds: khoảng thời gian tối đa (giây)
        Returns:
            True nếu person được thấy gần đây trong time window
        """
        time_diff = (current_time - self.last_seen).total_seconds()
        return time_diff <= time_window_seconds
    
    def get_time_since_last_seen(self, current_time):
        """Lấy thời gian kể từ lần cuối nhìn thấy (giây)"""
        return (current_time - self.last_seen).total_seconds()
    
    def is_feasible_transition(self, current_camera, camera_topology, camera_transition_max_time, current_time):
        """
        Kiểm tra xem việc chuyển từ last_camera sang current_camera có khả thi không
        
        Args:
            current_camera: camera hiện tại phát hiện
            camera_topology: dict định nghĩa kết nối giữa cameras
            camera_transition_max_time: dict định nghĩa thời gian transition tối đa
            current_time: thời điểm hiện tại
            
        Returns:
            tuple (is_feasible: bool, reason: str)
        """
        time_diff = self.get_time_since_last_seen(current_time)
        
        # Rule 1: Same camera → always allow (detector drop, occlusion, standing still)
        if current_camera == self.last_camera:
            return (True, f"same_camera (cam {current_camera})")
        
        # Rule 2: Check topology-based transition
        if self.last_camera in camera_topology:
            connected_cameras = camera_topology[self.last_camera]
            
            if current_camera in connected_cameras:
                # Cameras are connected - check transition time
                transition_key = f"{self.last_camera}->{current_camera}"
                
                if transition_key in camera_transition_max_time:
                    max_time = camera_transition_max_time[transition_key]
                    
                    if time_diff <= max_time:
                        return (True, f"topology_transition ({transition_key}, Δt={time_diff:.2f}s <= {max_time}s)")
                    else:
                        return (False, f"topology_timeout ({transition_key}, Δt={time_diff:.2f}s > {max_time}s)")
                else:
                    # No explicit max time defined, but cameras are connected
                    # Fall through to time window check
                    pass
            else:
                # Cameras not physically connected
                return (False, f"topology_blocked (cam {self.last_camera} -> {current_camera} not connected)")
        
        # Rule 3: Fallback to anti-reuse time window
        # This prevents old IDs from being reused by new people
        return (True, f"time_window_fallback (Δt={time_diff:.2f}s)") if time_diff <= 999999 else (False, "unknown")
    
    def get_summary(self):
        """Lấy thông tin tổng hợp"""
        return {
            'person_id': self.person_id,
            'state': self.state.value,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'duration_seconds': round(self.get_duration(), 2),
            'cameras_visited': list(self.cameras_visited.keys()),
            'camera_visit_counts': self.cameras_visited,
            'total_detections': self.total_detections,
            'avg_confidence': round(sum(self.confidences) / len(self.confidences), 2),
            'max_frames_missing': self.max_frames_missing,
            'state_transitions': len(self.state_history) - 1
        }
    
    def is_active(self):
        """Kiểm tra person có đang active không"""
        return self.state in [PersonState.DETECTED, PersonState.TRACKING]
    
    def should_confirm_lost(self, max_missing_frames=90):
        """Kiểm tra có nên confirm lost không"""
        return self.state == PersonState.LOST and self.frames_missing >= max_missing_frames
    
    def should_archive(self, min_inactive_seconds=300):
        """Kiểm tra có nên archive không (5 phút không hoạt động)"""
        if self.state != PersonState.CONFIRMED_LOST:
            return False
        inactive_time = (datetime.now() - self.last_seen).total_seconds()
        return inactive_time >= min_inactive_seconds


class PersonLifecycleManager:
    """Quản lý vòng đời của tất cả persons"""
    
    def __init__(self, output_dir="./tracking_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Quản lý persons theo state
        self.active_persons = {}      # DETECTED, TRACKING
        self.lost_persons = {}        # LOST
        self.archived_persons = {}    # CONFIRMED_LOST, ARCHIVED
        
        # Thống kê
        self.next_id = 0
        self.total_persons_seen = 0
        self.session_start = datetime.now()
        
        # Config
        self.max_lost_frames = 30        # Frames để chuyển TRACKING -> LOST
        self.max_confirm_lost_frames = 90  # Frames để chuyển LOST -> CONFIRMED_LOST
        self.archive_after_seconds = 300   # Giây để chuyển CONFIRMED_LOST -> ARCHIVED
        
        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Time window matching statistics
        self.time_window_rejections = 0
        self.topology_rejections = 0
        self.same_camera_matches = 0
        self.topology_transitions = 0
    
    def create_person(self, camera_id, confidence, bbox, match_info=None):
        """
        Tạo person mới
        
        Args:
            camera_id: ID của camera
            confidence: Độ tin cậy
            bbox: Bounding box
            match_info: Dict chứa metadata (cho person mới thường là None hoặc rejection info)
        
        Return: person_id
        """
        person_id = self.next_id
        person = PersonLifecycle(person_id, camera_id, confidence, bbox)
        
        # Nếu có match_info (thường là rejection reasoning), cập nhật detection đầu tiên
        if match_info:
            person.detections_history[-1].update({
                'match_score': match_info.get('match_score'),
                'matched_global_id': match_info.get('matched_global_id'),
                'match_confidence': match_info.get('match_confidence'),
                'reasoning': match_info.get('reasoning'),
                'feasibility_reason': match_info.get('feasibility_reason')
            })
        
        self.active_persons[person_id] = person
        self.total_persons_seen += 1
        self.next_id += 1
        
        print(f"\n✨ NEW PERSON: ID {person_id} | Camera {camera_id} | Confidence {confidence:.2f}")
        if match_info and match_info.get('reasoning'):
            print(f"   📝 Reason: {match_info['reasoning']}")
        
        return person_id
    
    def update_person(self, person_id, camera_id, confidence, bbox, match_info=None):
        """
        Cập nhật person khi phát hiện
        
        Args:
            person_id: ID của person
            camera_id: ID của camera
            confidence: Độ tin cậy
            bbox: Bounding box
            match_info: Dict chứa metadata về matching
        """
        
        # Tìm person trong active hoặc lost
        if person_id in self.active_persons:
            self.active_persons[person_id].update(camera_id, confidence, bbox, match_info)
        elif person_id in self.lost_persons:
            person = self.lost_persons[person_id]
            person.update(camera_id, confidence, bbox, match_info)
            # Di chuyển về active
            self.active_persons[person_id] = person
            del self.lost_persons[person_id]
        else:
            print(f"⚠️  Warning: Person {person_id} không tìm thấy trong active/lost")
    
    def get_matchable_persons_topology(self, current_camera, current_time, time_window_seconds, 
                                        camera_topology, camera_transition_max_time):
        """
        Lấy danh sách persons có thể match dựa trên topology và time constraints
        
        Args:
            current_camera: camera hiện tại
            current_time: datetime object
            time_window_seconds: float - time window fallback
            camera_topology: dict - camera connectivity
            camera_transition_max_time: dict - max transition times
            
        Returns:
            dict of {person_id: (person, feasibility_reason)}
        """
        matchable = {}
        
        # Kiểm tra active persons
        for person_id, person in self.active_persons.items():
            is_feasible, reason = person.is_feasible_transition(
                current_camera, 
                camera_topology, 
                camera_transition_max_time,
                current_time
            )
            
            if is_feasible:
                matchable[person_id] = (person, reason)
        
        # Kiểm tra lost persons (có thể tìm lại thông qua topology)
        for person_id, person in self.lost_persons.items():
            is_feasible, reason = person.is_feasible_transition(
                current_camera,
                camera_topology,
                camera_transition_max_time, 
                current_time
            )
            
            if is_feasible:
                matchable[person_id] = (person, reason)
        
        return matchable
    
    def get_matchable_persons(self, current_time, time_window_seconds):
        """
        Lấy danh sách persons có thể match (trong time window)
        Args:
            current_time: datetime object
            time_window_seconds: float
        Returns:
            dict of {person_id: person} trong time window
        """
        matchable = {}
        
        # Kiểm tra active persons
        for person_id, person in self.active_persons.items():
            if person.is_within_time_window(current_time, time_window_seconds):
                matchable[person_id] = person
        
        # Kiểm tra lost persons (có thể tìm lại)
        for person_id, person in self.lost_persons.items():
            if person.is_within_time_window(current_time, time_window_seconds):
                matchable[person_id] = person
        
        return matchable
    
    def process_frame_end(self, detected_ids):
        """
        Gọi sau mỗi frame để update lifecycle
        detected_ids: List các person_id được phát hiện trong frame này
        """
        detected_set = set(detected_ids)
        
        # Đánh dấu những person không được phát hiện
        for person_id, person in list(self.active_persons.items()):
            if person_id not in detected_set:
                person.mark_missing()
                
                # Chuyển sang LOST nếu cần
                if person.state == PersonState.LOST:
                    self.lost_persons[person_id] = person
                    del self.active_persons[person_id]
        
        # Kiểm tra lost persons
        for person_id, person in list(self.lost_persons.items()):
            person.mark_missing()
            
            # Confirm lost nếu mất quá lâu
            if person.should_confirm_lost(self.max_confirm_lost_frames):
                person.confirm_lost()
                self.archived_persons[person_id] = person
                del self.lost_persons[person_id]
        
        # Archive confirmed lost persons
        for person_id, person in list(self.archived_persons.items()):
            if person.state == PersonState.CONFIRMED_LOST:
                if person.should_archive(self.archive_after_seconds):
                    person.archive()
    
    def get_all_persons(self):
        """Lấy tất cả persons"""
        all_persons = {}
        all_persons.update(self.active_persons)
        all_persons.update(self.lost_persons)
        all_persons.update(self.archived_persons)
        return all_persons
    
    def get_statistics(self):
        """Lấy thống kê tổng quan"""
        all_persons = self.get_all_persons()
        
        state_counts = defaultdict(int)
        for person in all_persons.values():
            state_counts[person.state.value] += 1
        
        return {
            'session_id': self.session_id,
            'session_duration': (datetime.now() - self.session_start).total_seconds(),
            'total_persons': self.total_persons_seen,
            'active_persons': len(self.active_persons),
            'lost_persons': len(self.lost_persons),
            'archived_persons': len(self.archived_persons),
            'state_distribution': dict(state_counts),
            'time_window_rejections': self.time_window_rejections,
            'topology_rejections': self.topology_rejections,
            'same_camera_matches': self.same_camera_matches,
            'topology_transitions': self.topology_transitions
        }
    
    def print_status(self):
        """In trạng thái hiện tại"""
        stats = self.get_statistics()
        
        print("\n" + "="*80)
        print(f"LIFECYCLE STATUS - {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)
        print(f"🟢 Active: {stats['active_persons']} | 🟡 Lost: {stats['lost_persons']} | 🔴 Archived: {stats['archived_persons']}")
        print(f"📊 Total seen: {stats['total_persons']} | ⏱️  Session: {stats['session_duration']:.1f}s")
        print(f"⏰ Time window rejections: {stats['time_window_rejections']}")
        print(f"🚫 Topology rejections: {stats['topology_rejections']}")
        print(f"📹 Same camera: {stats['same_camera_matches']} | 🔄 Topology transitions: {stats['topology_transitions']}")
        print("-"*80)
    
    def save_summary(self):
        """Lưu tổng kết"""
        all_persons = self.get_all_persons()
        
        # Chuẩn bị dữ liệu
        data = {
            'session': self.get_statistics(),
            'persons': [p.get_summary() for p in all_persons.values()]
        }
        
        # Lưu JSON
        json_file = self.output_dir / f"lifecycle_{self.session_id}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved: {json_file}")
        
        # Lưu CSV
        csv_file = self.output_dir / f"lifecycle_{self.session_id}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'person_id', 'state', 'first_seen', 'last_seen', 
                'duration_seconds', 'cameras_visited', 'total_detections',
                'avg_confidence', 'max_frames_missing'
            ])
            writer.writeheader()
            for person in all_persons.values():
                summary = person.get_summary()
                summary['cameras_visited'] = ','.join(map(str, summary['cameras_visited']))
                del summary['camera_visit_counts']
                del summary['state_transitions']
                writer.writerow(summary)
        print(f"✅ Saved: {csv_file}")
        
        # Lưu chi tiết detections với match metadata
        detections_file = self.output_dir / f"detections_{self.session_id}.csv"
        with open(detections_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'person_id', 'timestamp', 'camera_id', 'confidence', 'state',
                'match_score', 'matched_global_id', 'match_confidence', 
                'reasoning', 'feasibility_reason'
            ])
            writer.writeheader()
            for person in all_persons.values():
                for detection in person.detections_history:
                    row = {
                        'person_id': person.person_id,
                        'timestamp': detection['timestamp'],
                        'camera_id': detection['camera_id'],
                        'confidence': detection['confidence'],
                        'state': detection['state'],
                        'match_score': detection.get('match_score', ''),
                        'matched_global_id': detection.get('matched_global_id', ''),
                        'match_confidence': detection.get('match_confidence', ''),
                        'reasoning': detection.get('reasoning', ''),
                        'feasibility_reason': detection.get('feasibility_reason', '')
                    }
                    writer.writerow(row)
        print(f"✅ Saved: {detections_file}")
    
    def print_final_report(self):
        """In báo cáo cuối cùng"""
        all_persons = self.get_all_persons()
        stats = self.get_statistics()
        
        print("\n" + "="*80)
        print(f"FINAL LIFECYCLE REPORT - Session: {self.session_id}")
        print("="*80)
        print(f"⏱️  Session duration: {stats['session_duration']:.1f} seconds")
        print(f"👥 Total persons: {stats['total_persons']}")
        print(f"📊 State distribution: {stats['state_distribution']}")
        print(f"⏰ Time window rejections: {stats['time_window_rejections']}")
        print(f"🚫 Topology rejections: {stats['topology_rejections']}")
        print(f"📹 Same camera matches: {stats['same_camera_matches']}")
        print(f"🔄 Topology transitions: {stats['topology_transitions']}")
        print("-"*80)
        
        for person in sorted(all_persons.values(), key=lambda p: p.person_id):
            summary = person.get_summary()
            state_icon = {
                'detected': '🆕',
                'tracking': '🟢',
                'lost': '🟡',
                'confirmed_lost': '🔴',
                'archived': '📦'
            }
            
            print(f"\n{state_icon[summary['state']]} Person {summary['person_id']} [{summary['state'].upper()}]")
            print(f"   ⏰ {summary['first_seen']} -> {summary['last_seen']}")
            print(f"   ⏳ Duration: {summary['duration_seconds']}s")
            print(f"   🎥 Cameras: {summary['cameras_visited']}")
            print(f"   🔍 Detections: {summary['total_detections']}")
            print(f"   ✓ Avg confidence: {summary['avg_confidence']}")
            print(f"   ⚠️  Max missing: {summary['max_frames_missing']} frames")
        
        print("\n" + "="*80)