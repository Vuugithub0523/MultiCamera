"""
Multi-Camera Manager
Manages multiple camera pipelines with RTSP and lifecycle management
"""
import asyncio
from typing import Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

from config import Config
from detection import YOLODetector, BYTETracker
from reid import FeatureExtractor, PersonDatabase
from .rtsp_loader import RTSPStreamLoader
from .pipeline import CameraPipeline
from .lifecycle_manager import PersonLifecycleManager


class MultiCameraManager:
    """Manager for multiple camera processing pipelines with lifecycle tracking"""
    
    def __init__(self, config: Config):
        """
        Initialize multi-camera manager
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Shared models (loaded once, used by all cameras)
        print("[Manager] Loading shared models...")
        self.detector = YOLODetector(
            model_path=config.YOLO_MODEL_PATH,
            coco_names_path=config.COCO_NAMES_PATH,
            device=config.DEVICE,
            confidence_threshold=config.DETECTION_CONFIDENCE,
            target_classes=["person"]
        )
        
        self.feature_extractor = FeatureExtractor(
            model_path=config.REID_MODEL_PATH,
            device=config.DEVICE
        )
        
        self.person_db = PersonDatabase(
            db_file=config.PERSON_DB_FILE,
            max_gallery_size=config.MAX_GALLERY_SIZE
        )
        
        # Lifecycle manager (shared across all cameras)
        self.lifecycle_manager = PersonLifecycleManager(
            output_dir=config.TRACKING_LOG_DIR
        )
        self.lifecycle_manager.max_lost_frames = config.MAX_LOST_FRAMES
        self.lifecycle_manager.max_confirm_lost_frames = config.MAX_CONFIRM_LOST_FRAMES
        self.lifecycle_manager.archive_after_seconds = config.ARCHIVE_AFTER_SECONDS
        
        # Per-camera components
        self.loaders: Dict[str, RTSPStreamLoader] = {}
        self.trackers: Dict[str, BYTETracker] = {}
        self.pipelines: Dict[str, CameraPipeline] = {}
        
        # Processing control
        self.running = False
        self.tasks = []
        
        # Thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=len(config.get_enabled_cameras()))
        
        # Initialize cameras
        self._initialize_cameras()
    
    def _initialize_cameras(self):
        """Initialize RTSP loaders, trackers, and pipelines"""
        cameras = self.config.get_enabled_cameras()
        
        for cam in cameras:
            camera_id = cam['id']
            rtsp_url = cam['rtsp_url']
            
            # Create RTSP stream loader
            loader = RTSPStreamLoader(
                url=rtsp_url,
                name=camera_id,
                buffer_size=1,  # Minimum latency
                target_width=self.config.INPUT_WIDTH,
                target_height=self.config.INPUT_HEIGHT
            ).start()
            self.loaders[camera_id] = loader
            
            # Create tracker (per-camera instance)
            tracker = BYTETracker(
                track_thresh=self.config.TRACK_THRESH,
                match_thresh=self.config.MATCH_THRESH,
                track_buffer=self.config.TRACK_BUFFER,
                frame_rate=self.config.FRAME_RATE,
                mot20=False
            )
            self.trackers[camera_id] = tracker
            
            # Create pipeline with lifecycle manager and topology
            pipeline = CameraPipeline(
                camera_id=camera_id,
                detector=self.detector,
                tracker=tracker,
                feature_extractor=self.feature_extractor,
                person_db=self.person_db,
                lifecycle_manager=self.lifecycle_manager,
                detect_skip_frames=self.config.DETECTION_SKIP_FRAMES,
                output_fps=self.config.OUTPUT_FPS,
                reid_threshold=self.config.REID_THRESHOLD,
                time_window_seconds=self.config.TIME_WINDOW_SECONDS,
                camera_topology=getattr(self.config, 'CAMERA_TOPOLOGY', {}),
                camera_transition_max_time=getattr(self.config, 'CAMERA_TRANSITION_MAX_TIME', {})
            )
            self.pipelines[camera_id] = pipeline
            
            print(f"[Manager] Initialized camera: {camera_id}")
        
        print(f"[Manager] All cameras initialized with lifecycle tracking")
    
    def start_all(self):
        """Start all camera stream loaders"""
        self.running = True
        print(f"[Manager] Started {len(self.loaders)} camera streams")
    
    def stop_all(self):
        """Stop all camera loaders and processing"""
        self.running = False
        
        for loader in self.loaders.values():
            loader.stop()
        
        for pipeline in self.pipelines.values():
            pipeline.cleanup()
        
        self.executor.shutdown(wait=True)
        
        # Save person database
        self.person_db.save()
        
        # Export lifecycle summary
        self.lifecycle_manager.export_summary_csv()
        
        print("[Manager] Stopped all cameras")
    
    async def process_camera_loop(self, camera_id: str, frame_callback: Optional[Callable] = None):
        """
        Main processing loop for a single camera
        
        Args:
            camera_id: Camera identifier
            frame_callback: Optional callback function(camera_id, jpeg_bytes, tracks)
        """
        loader = self.loaders.get(camera_id)
        pipeline = self.pipelines.get(camera_id)
        
        if not loader or not pipeline:
            print(f"[Manager] Camera {camera_id} not found")
            return
        
        print(f"[Manager] Starting processing loop for {camera_id}")
        
        while self.running:
            try:
                # Get frame from loader
                frame, timestamp = loader.read()
                
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                
                # Process frame through pipeline
                jpeg_bytes, tracks = await pipeline.process_frame(frame)
                
                # Call callback if provided and we have output
                if frame_callback and jpeg_bytes is not None:
                    await frame_callback(camera_id, jpeg_bytes, tracks)
                
            except Exception as e:
                print(f"[Manager] Error processing {camera_id}: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(0.1)
        
        print(f"[Manager] Processing loop ended for {camera_id}")
    
    async def start_processing(self, frame_callback: Optional[Callable] = None):
        """
        Start processing all cameras
        
        Args:
            frame_callback: Optional callback for processed frames
        """
        if not self.running:
            self.start_all()
        
        # Create tasks for all cameras
        self.tasks = [
            asyncio.create_task(self.process_camera_loop(camera_id, frame_callback))
            for camera_id in self.loaders.keys()
        ]
        
        print(f"[Manager] Started processing {len(self.tasks)} camera streams")
        
        # Wait for all tasks
        await asyncio.gather(*self.tasks, return_exceptions=True)
    
    def get_pipeline(self, camera_id: str) -> Optional[CameraPipeline]:
        """Get pipeline for a specific camera"""
        return self.pipelines.get(camera_id)
    
    def get_stats(self) -> Dict:
        """Get statistics for all cameras and lifecycle"""
        stats = {
            'cameras': {},
            'person_db': self.person_db.get_stats(),
            'lifecycle': self.lifecycle_manager.get_stats()
        }
        
        for camera_id in self.loaders.keys():
            loader_stats = {
                'is_opened': self.loaders[camera_id].is_opened(),
                'frame_count': self.loaders[camera_id].frame_count,
                'fps': round(self.loaders[camera_id].fps, 1)
            }
            stats['cameras'][camera_id] = {
                'loader': loader_stats,
                'pipeline': self.pipelines[camera_id].get_stats()
            }
        
        return stats
    
    def get_camera_ids(self) -> List[str]:
        """Get list of camera IDs"""
        return list(self.loaders.keys())
