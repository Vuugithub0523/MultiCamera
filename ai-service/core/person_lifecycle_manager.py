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
        """Thêm một detection vào lịch sử"""
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
    
    def update(self, camera_id, confidence, bbox, match_info=None):
        """Cập nhật khi phát hiện person"""
        now = datetime.now()
        
        if self.frames_missing > 0:
            self.max_frames_missing = max(self.max_frames_missing, self.frames_missing)
            self.frames_missing = 0
        
        old_state = self.state
        if self.state == PersonState.DETECTED:
            self.state = PersonState.TRACKING
            self._add_state_change(old_state, PersonState.TRACKING)
        elif self.state == PersonState.LOST:
            self.state = PersonState.TRACKING
            self._add_state_change(old_state, PersonState.TRACKING)
        
        self.last_seen = now
        self.last_camera = self.current_camera
        self.current_camera = camera_id
        self.total_detections += 1
        self.confidences.append(confidence)
        
        if camera_id not in self.camera_history or self.camera_history[-1] != camera_id:
            self.camera_history.append(camera_id)
        
        if camera_id in self.cameras_visited:
            self.cameras_visited[camera_id] += 1
        else:
            self.cameras_visited[camera_id] = 1
        
        self._add_detection(camera_id, confidence, bbox, match_info)
    
    def mark_missing(self):
        """Đánh dấu person không được phát hiện trong frame hiện tại"""
        self.frames_missing += 1
        
        if self.frames_missing == 30 and self.state == PersonState.TRACKING:
            old_state = self.state
            self.state = PersonState.LOST
            self._add_state_change(old_state, PersonState.LOST)
    
    def confirm_lost(self):
        """Xác nhận person đã rời đi hẳn"""
        if self.state == PersonState.LOST:
            old_state = self.state
            self.state = PersonState.CONFIRMED_LOST
            self._add_state_change(old_state, PersonState.CONFIRMED_LOST)
            return True
        return False
    
    def archive(self):
        """Lưu trữ person"""
        if self.state == PersonState.CONFIRMED_LOST:
            old_state = self.state
            self.state = PersonState.ARCHIVED
            self._add_state_change(old_state, PersonState.ARCHIVED)
            return True
        return False
    
    def _add_state_change(self, old_state, new_state):
        """Ghi lại sự thay đổi state"""
        self.state_history.append((new_state, datetime.now()))
    
    def get_duration(self):
        """Tính tổng thời gian (giây)"""
        return (self.last_seen - self.first_seen).total_seconds()
    
    def is_within_time_window(self, current_time, time_window_seconds):
        """Kiểm tra xem person có nằm trong time window không"""
        time_diff = (current_time - self.last_seen).total_seconds()
        return time_diff <= time_window_seconds
    
    def get_time_since_last_seen(self, current_time):
        """Lấy thời gian kể từ lần cuối nhìn thấy (giây)"""
        return (current_time - self.last_seen).total_seconds()
    
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
        """Kiểm tra có nên archive không"""
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
        self.max_lost_frames = 30
        self.max_confirm_lost_frames = 90
        self.archive_after_seconds = 300
        
        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Statistics
        self.time_window_rejections = 0
        self.topology_rejections = 0
        self.same_camera_matches = 0
        self.topology_transitions = 0
        
        # Event tracking for real-time notifications
        self.recent_events = []
        self.max_recent_events = 100
    
    def create_person(self, camera_id, confidence, bbox, match_info=None):
        """Tạo person mới"""
        person_id = self.next_id
        person = PersonLifecycle(person_id, camera_id, confidence, bbox)
        
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
        
        # Add event
        self._add_event('appear', person_id, camera_id, confidence)
        
        return person_id
    
    def update_person(self, person_id, camera_id, confidence, bbox, match_info=None):
        """Cập nhật person khi phát hiện"""
        if person_id in self.active_persons:
            old_camera = self.active_persons[person_id].current_camera
            self.active_persons[person_id].update(camera_id, confidence, bbox, match_info)
            # Add event if moved to different camera
            if old_camera != camera_id:
                self._add_event('move', person_id, camera_id, confidence, old_camera)
        elif person_id in self.lost_persons:
            person = self.lost_persons[person_id]
            old_camera = person.current_camera
            person.update(camera_id, confidence, bbox, match_info)
            self.active_persons[person_id] = person
            del self.lost_persons[person_id]
            # Add re-appear event
            if old_camera != camera_id:
                self._add_event('move', person_id, camera_id, confidence, old_camera)
            else:
                self._add_event('reappear', person_id, camera_id, confidence)
    
    def get_matchable_persons(self, current_time, time_window_seconds):
        """Lấy danh sách persons có thể match (trong time window)"""
        matchable = {}
        
        for person_id, person in self.active_persons.items():
            if person.is_within_time_window(current_time, time_window_seconds):
                matchable[person_id] = person
        
        for person_id, person in self.lost_persons.items():
            if person.is_within_time_window(current_time, time_window_seconds):
                matchable[person_id] = person
        
        return matchable
    
    def process_frame_end(self, detected_ids):
        """Gọi sau mỗi frame để update lifecycle"""
        detected_set = set(detected_ids)
        
        for person_id, person in list(self.active_persons.items()):
            if person_id not in detected_set:
                person.mark_missing()
                
                if person.state == PersonState.LOST:
                    self.lost_persons[person_id] = person
                    del self.active_persons[person_id]
        
        for person_id, person in list(self.lost_persons.items()):
            person.mark_missing()
            
            if person.should_confirm_lost(self.max_confirm_lost_frames):
                person.confirm_lost()
                self.archived_persons[person_id] = person
                del self.lost_persons[person_id]
        
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
    
    def save_summary(self):
        """Lưu tổng kết"""
        all_persons = self.get_all_persons()
        
        data = {
            'session': self.get_statistics(),
            'persons': [p.get_summary() for p in all_persons.values()]
        }
        
        json_file = self.output_dir / f"lifecycle_{self.session_id}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        csv_file = self.output_dir / f"lifecycle_{self.session_id}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if all_persons:
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
    
    def print_final_report(self):
        """In báo cáo cuối cùng"""
        stats = self.get_statistics()
        
        print(f"\n{'='*80}")
        print(f"FINAL REPORT - Session: {self.session_id}")
        print(f"{'='*80}")
        print(f"Total persons: {stats['total_persons']}")
        print(f"Duration: {stats['session_duration']:.1f}s")
        print(f"State distribution: {stats['state_distribution']}")
        print(f"{'='*80}\n")    
    def _add_event(self, event_type, person_id, camera_id, confidence, from_camera=None):
        """Thêm sự kiện mới"""
        event = {
            'id': len(self.recent_events) + 1,
            'timestamp': datetime.now().isoformat(),
            'time': datetime.now().strftime("%H:%M:%S"),
            'type': event_type,
            'person_id': person_id,
            'camera_id': camera_id,
            'confidence': round(confidence * 100, 1)
        }
        if from_camera:
            event['from_camera'] = from_camera
        
        self.recent_events.append(event)
        # Keep only last N events
        if len(self.recent_events) > self.max_recent_events:
            self.recent_events.pop(0)
    
    def get_recent_events(self, limit=50):
        """Lấy các sự kiện gần đây"""
        return self.recent_events[-limit:] if limit else self.recent_events
    
    def get_hourly_traffic(self):
        """Lấy số lượng người theo giờ"""
        hourly_counts = defaultdict(int)
        for person in self.get_all_persons().values():
            hour = person.first_seen.strftime("%H:00")
            hourly_counts[hour] += 1
        
        # Generate 24-hour data
        result = []
        for h in range(24):
            hour_str = f"{h:02d}:00"
            result.append({
                'hour': hour_str,
                'count': hourly_counts.get(hour_str, 0)
            })
        return result
    
    def get_camera_flow(self):
        """Lấy luồng di chuyển giữa các camera"""
        camera_transitions = defaultdict(int)
        camera_totals = defaultdict(int)
        
        for person in self.get_all_persons().values():
            # Count total people per camera
            for cam in person.cameras_visited.keys():
                camera_totals[cam] += 1
            
            # Count transitions
            for i in range(len(person.camera_history) - 1):
                from_cam = person.camera_history[i]
                to_cam = person.camera_history[i + 1]
                if from_cam != to_cam:
                    key = f"{from_cam}->{to_cam}"
                    camera_transitions[key] += 1
        
        return {
            'totals': dict(camera_totals),
            'transitions': dict(camera_transitions)
        }