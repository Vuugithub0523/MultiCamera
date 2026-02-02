"""
AI Service Configuration Module
Loads and validates configuration from config.yaml
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class CameraConfig:
    id: str
    rtsp: str


@dataclass
class ModelConfig:
    detector_onnx_path: str
    classes_path: str


@dataclass
class ReIDConfig:
    enabled: bool
    feature_onnx_path: str
    similarity_threshold: float


@dataclass
class AIConfig:
    enabled: bool
    frame_interval: int
    det_conf: float
    resize_for_inference: float
    model: ModelConfig
    reid: ReIDConfig


@dataclass
class AnnotateConfig:
    draw_boxes: bool
    draw_labels: bool
    draw_fps: bool
    draw_latency: bool
    box_color: Tuple[int, int, int]
    text_color: Tuple[int, int, int]
    box_thickness: int
    font_scale: float


@dataclass
class RestreamConfig:
    rtsp_base: str
    fps: int
    resolution: Tuple[int, int]
    codec: str
    preset: str
    tune: str
    bitrate: str


@dataclass
class LiveKitConfig:
    url: str
    api_key: str
    api_secret: str
    room: str


@dataclass
class APIConfig:
    host: str
    port: int


@dataclass
class LoggingConfig:
    level: str
    dir: str


@dataclass
class AppConfig:
    cameras: List[CameraConfig]
    ai: AIConfig
    annotate: AnnotateConfig
    restream: RestreamConfig
    livekit: LiveKitConfig
    api: APIConfig
    logging: LoggingConfig
    project_root: Path = field(default_factory=Path)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from YAML file."""
    if config_path is None:
        # Default: look for config.yaml in project root
        config_path = os.environ.get("CONFIG_PATH", "../config.yaml")
    
    # Resolve relative to ai-service directory
    ai_service_dir = Path(__file__).parent.parent
    config_file = (ai_service_dir / config_path).resolve()
    
    if not config_file.exists():
        # Try from project root
        project_root = ai_service_dir.parent
        config_file = project_root / "config.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    
    project_root = config_file.parent
    
    # Parse cameras
    cameras = [
        CameraConfig(id=c["id"], rtsp=c["rtsp"])
        for c in raw.get("cameras", [])
    ]
    
    # Parse AI config
    ai_raw = raw.get("ai", {})
    model_raw = ai_raw.get("model", {})
    reid_raw = ai_raw.get("reid", {})
    
    ai = AIConfig(
        enabled=ai_raw.get("enabled", True),
        frame_interval=ai_raw.get("frame_interval", 3),
        det_conf=ai_raw.get("det_conf", 0.7),
        resize_for_inference=ai_raw.get("resize_for_inference", 0.5),
        model=ModelConfig(
            detector_onnx_path=str(project_root / model_raw.get("detector_onnx_path", "./models/pretrained_models/yolov4-tiny.onnx")),
            classes_path=str(project_root / model_raw.get("classes_path", "./models/pretrained_models/coco.names")),
        ),
        reid=ReIDConfig(
            enabled=reid_raw.get("enabled", False),
            feature_onnx_path=str(project_root / reid_raw.get("feature_onnx_path", "./models/pretrained_models/osnet_ain_x1_0_M.onnx")),
            similarity_threshold=reid_raw.get("similarity_threshold", 0.42),
        ),
    )
    
    # Parse annotate config
    ann_raw = raw.get("annotate", {})
    annotate = AnnotateConfig(
        draw_boxes=ann_raw.get("draw_boxes", True),
        draw_labels=ann_raw.get("draw_labels", True),
        draw_fps=ann_raw.get("draw_fps", True),
        draw_latency=ann_raw.get("draw_latency", True),
        box_color=tuple(ann_raw.get("box_color", [0, 255, 0])),
        text_color=tuple(ann_raw.get("text_color", [255, 255, 255])),
        box_thickness=ann_raw.get("box_thickness", 2),
        font_scale=ann_raw.get("font_scale", 0.6),
    )
    
    # Parse restream config
    rs_raw = raw.get("restream", {})
    restream = RestreamConfig(
        rtsp_base=rs_raw.get("rtsp_base", "rtsp://127.0.0.1:8554"),
        fps=rs_raw.get("fps", 15),
        resolution=tuple(rs_raw.get("resolution", [1280, 720])),
        codec=rs_raw.get("codec", "h264"),
        preset=rs_raw.get("preset", "veryfast"),
        tune=rs_raw.get("tune", "zerolatency"),
        bitrate=rs_raw.get("bitrate", "2M"),
    )
    
    # Parse LiveKit config
    lk_raw = raw.get("livekit", {})
    livekit = LiveKitConfig(
        url=lk_raw.get("url", "ws://127.0.0.1:7880"),
        api_key=lk_raw.get("api_key", "devkey"),
        api_secret=lk_raw.get("api_secret", "devsecret"),
        room=lk_raw.get("room", "multicam"),
    )
    
    # Parse API config
    api_raw = raw.get("api", {})
    api = APIConfig(
        host=api_raw.get("host", "0.0.0.0"),
        port=api_raw.get("port", 8080),
    )
    
    # Parse logging config
    log_raw = raw.get("logging", {})
    logging_cfg = LoggingConfig(
        level=log_raw.get("level", "INFO"),
        dir=str(project_root / log_raw.get("dir", "./logs")),
    )
    
    return AppConfig(
        cameras=cameras,
        ai=ai,
        annotate=annotate,
        restream=restream,
        livekit=livekit,
        api=api,
        logging=logging_cfg,
        project_root=project_root,
    )


# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config() -> AppConfig:
    """Reload configuration from file."""
    global _config
    _config = load_config()
    return _config
