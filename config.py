"""
Configuration for Native AI Backend
Realtime multi-camera tracking system with RTSP input
"""
import os
from typing import List, Dict

class Config:
    """Main configuration class"""
    
    # ============================================
    # CAMERAS CONFIGURATION
    # ============================================
    # NOTE: Change USE_VIDEO_FILES to True to test with video files instead of RTSP
    USE_VIDEO_FILES = False  # Set to True for demo/testing
    
    CAMERAS: List[Dict[str, str]] = [
        {
            "id": "cam01",
            "name": "Camera 01 - Entrance",
            "rtsp_url": "rtsp://admin:12345678Chuong@192.168.1.204:554/cam/realmonitor?channel=1&subtype=0&tcp",
            "enabled": True,
        },
        {
            "id": "cam02",
            "name": "Camera 02 - Lobby",
            "rtsp_url": "rtsp://admin:123456Chuong@192.168.1.253:554/cam/realmonitor?channel=1&subtype=0&tcp",
            "enabled": True,  # Enabled - camera is now active
        },
        {
            "id": "cam03",
            "name": "Camera 03 - Warehouse",
            "rtsp_url": "rtsp://admin:1234567Chuong@192.168.1.254:554/cam/realmonitor?channel=1&subtype=0&tcp",
            "enabled": True,
        },
    ]
    
    # ============================================
    # MODEL PATHS
    # ============================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    # YOLO Detection Model
    YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov4-tiny.onnx")
    COCO_NAMES_PATH = os.path.join(MODELS_DIR, "coco.names")
    
    # Re-ID Feature Extraction Model
    REID_MODEL_PATH = os.path.join(MODELS_DIR, "osnet_ain_x1_0_M.onnx")
    
    # Device: "cuda" or "cpu"
    DEVICE = "cuda"  # Change to "cpu" if no GPU
    
    # ============================================
    # DETECTION & TRACKING SETTINGS
    # ============================================
    # YOLO Detection
    DETECTION_CONFIDENCE = 0.25  # Lowered from 0.5 for better detection
    DETECTION_SKIP_FRAMES = 2  # Detect every N frames (1=every frame, 2=skip 1)
    
    # BYTETracker Settings
    TRACK_THRESH = 0.25         # Lowered from 0.5 - High confidence detection threshold
    MATCH_THRESH = 0.8          # IoU threshold for matching
    TRACK_BUFFER = 30           # Frames to keep lost tracks
    FRAME_RATE = 30             # Video frame rate
    MIN_BOX_AREA = 10           # Lowered from 100 - Minimum bounding box area
    
    # Re-ID Settings
    REID_THRESHOLD = 0.42       # Cosine distance threshold for same person
    MAX_GALLERY_SIZE = 512      # Max stored features per person
    
    # ============================================
    # LIFECYCLE MANAGEMENT
    # ============================================
    # Tracking lifecycle parameters
    MAX_LOST_FRAMES = 30           # Frames before marking as LOST
    MAX_CONFIRM_LOST_FRAMES = 90   # Frames before CONFIRMED_LOST
    ARCHIVE_AFTER_SECONDS = 300    # Seconds before archiving (5 minutes)
    
    # Time window for person matching
    TIME_WINDOW_SECONDS = 3.0      # Match persons within this time window
    
    # ============================================
    # FRAME PROCESSING
    # ============================================
    # Input frame settings
    INPUT_WIDTH = 1280
    INPUT_HEIGHT = 720
    
    # RTSP buffer settings
    RTSP_BUFFER_SIZE = 1        # Minimize latency (1-2 frames)
    RTSP_TIMEOUT = 10           # Seconds to wait for frame
    
    # Output JPEG quality for WebSocket
    JPEG_QUALITY = 80           # 1-100, higher = better quality
    OUTPUT_FPS = 15             # FPS to send to frontend (lower = less bandwidth)
    
    # ============================================
    # SERVER SETTINGS
    # ============================================
    HOST = "0.0.0.0"
    PORT = 5000
    
    # WebSocket settings
    WS_MAX_QUEUE_SIZE = 5       # Max frames queued per client
    WS_PING_INTERVAL = 20       # Seconds between ping/pong
    WS_PING_TIMEOUT = 10        # Seconds to wait for pong
    
    # ============================================
    # STORAGE
    # ============================================
    STORAGE_DIR = os.path.join(BASE_DIR, "storage")
    PERSON_DB_FILE = os.path.join(STORAGE_DIR, "persons.json")
    EVENTS_DIR = os.path.join(STORAGE_DIR, "events")
    TRACKING_LOG_DIR = os.path.join(STORAGE_DIR, "tracking_logs")  # Lifecycle logs
    
    # ============================================
    # LOGGING
    # ============================================
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # ============================================
    # PERFORMANCE TUNING (RTX 3050 8GB)
    # ============================================
    # Memory optimization
    MAX_FRAME_QUEUE_SIZE = 2    # Per camera
    CLEAR_OLD_TRACKS = True     # Remove tracks after TRACK_BUFFER frames
    
    # GPU batch processing (if multiple cameras)
    BATCH_DETECTION = False     # Set True if you want to batch 3 cameras
    
    @classmethod
    def get_enabled_cameras(cls):
        """Return only enabled cameras"""
        return [cam for cam in cls.CAMERAS if cam.get("enabled", True)]
    
    @classmethod
    def get_camera_by_id(cls, camera_id: str):
        """Get camera config by ID"""
        for cam in cls.CAMERAS:
            if cam["id"] == camera_id:
                return cam
        return None


# Development/Testing overrides
if os.getenv("ENV") == "development":
    Config.LOG_LEVEL = "DEBUG"
    Config.DETECTION_SKIP_FRAMES = 3  # Slower detection for debugging
    
# Use video files instead of RTSP (for testing without cameras)
if os.getenv("USE_VIDEO_FILES") == "1":
    VIDEO_DIR = os.getenv("VIDEO_DIR", "./videos")
    Config.CAMERAS = [
        {
            "id": "cam01",
            "name": "Camera 1",
            "rtsp_url": os.path.join(VIDEO_DIR, "video1.mp4"),
            "enabled": True,
        },
        {
            "id": "cam02",
            "name": "Camera 2",
            "rtsp_url": os.path.join(VIDEO_DIR, "video2.mp4"),
            "enabled": True,
        },
        {
            "id": "cam03",
            "name": "Camera 3",
            "rtsp_url": os.path.join(VIDEO_DIR, "video3.mp4"),
            "enabled": True,
        },
    ]
