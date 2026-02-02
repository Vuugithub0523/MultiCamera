"""
AI Service - Main Entry Point
FastAPI application with RTSP ingest, AI inference, annotation, and re-streaming.
"""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_config, AppConfig
from app.ingest.rtsp_ingest import RTSPIngestManager
from app.inference.detector import YOLOv4TinyDetector
from app.inference.reid import ReIDExtractor
from app.annotate.annotator import FrameAnnotator
from app.restream.ffmpeg_restreamer import FFmpegRestreamer
from app.events.event_manager import EventManager
from app.utils.logger import setup_logging


# Global components
config: AppConfig = None
ingest_manager: RTSPIngestManager = None
detector: YOLOv4TinyDetector = None
reid_extractor: ReIDExtractor = None
annotator: FrameAnnotator = None
restreamers: Dict[str, FFmpegRestreamer] = {}
event_manager: EventManager = None
pipeline_tasks: Dict[str, asyncio.Task] = {}
shutdown_event: asyncio.Event = None

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global config, ingest_manager, detector, reid_extractor, annotator
    global restreamers, event_manager, pipeline_tasks, shutdown_event
    
    # Startup
    logger.info("Starting AI Service...")
    
    config = get_config()
    setup_logging(config.logging.level, config.logging.dir)
    
    shutdown_event = asyncio.Event()
    
    # Initialize event manager
    event_manager = EventManager()
    
    # Initialize detector
    logger.info("Loading YOLOv4-tiny detector...")
    detector = YOLOv4TinyDetector(
        onnx_path=config.ai.model.detector_onnx_path,
        classes_path=config.ai.model.classes_path,
        confidence_threshold=config.ai.det_conf,
        device="cuda",
    )
    
    # Initialize ReID if enabled
    if config.ai.reid.enabled:
        logger.info("Loading ReID feature extractor...")
        reid_extractor = ReIDExtractor(
            onnx_path=config.ai.reid.feature_onnx_path,
            device="cuda",
        )
    
    # Initialize annotator
    annotator = FrameAnnotator(config.annotate)
    
    # Initialize RTSP ingest manager
    ingest_manager = RTSPIngestManager(config.cameras)
    
    # Initialize restreamers for each camera
    for cam in config.cameras:
        output_url = f"{config.restream.rtsp_base}/ann_{cam.id}"
        restreamers[cam.id] = FFmpegRestreamer(
            output_url=output_url,
            width=config.restream.resolution[0],
            height=config.restream.resolution[1],
            fps=config.restream.fps,
            preset=config.restream.preset,
            tune=config.restream.tune,
            bitrate=config.restream.bitrate,
        )
    
    # Start RTSP ingest
    await ingest_manager.start()
    
    # Start restreamers
    for cam_id, restreamer in restreamers.items():
        await restreamer.start()
        logger.info(f"Started restreamer for {cam_id}")
    
    # Start pipeline tasks for each camera
    for cam in config.cameras:
        task = asyncio.create_task(
            run_camera_pipeline(cam.id),
            name=f"pipeline_{cam.id}"
        )
        pipeline_tasks[cam.id] = task
        logger.info(f"Started pipeline for {cam.id}")
    
    logger.info("AI Service started successfully!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Service...")
    shutdown_event.set()
    
    # Cancel pipeline tasks
    for cam_id, task in pipeline_tasks.items():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    # Stop restreamers
    for restreamer in restreamers.values():
        await restreamer.stop()
    
    # Stop ingest
    await ingest_manager.stop()
    
    logger.info("AI Service stopped.")


async def run_camera_pipeline(cam_id: str):
    """Run the AI pipeline for a single camera."""
    frame_count = 0
    last_detections = []
    
    while not shutdown_event.is_set():
        try:
            # Get latest frame
            frame_data = await ingest_manager.get_frame(cam_id)
            if frame_data is None:
                await asyncio.sleep(0.01)
                continue
            
            frame, timestamp, fps_ingest = frame_data
            frame_count += 1
            
            # Skip frames based on interval
            if frame_count % config.ai.frame_interval != 0:
                # Use last detections for annotation
                detections = last_detections
            else:
                # Run inference
                if config.ai.enabled:
                    # Resize for inference if needed
                    inference_frame = frame
                    scale = config.ai.resize_for_inference
                    if scale < 1.0:
                        import cv2
                        h, w = frame.shape[:2]
                        inference_frame = cv2.resize(
                            frame, 
                            (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_LINEAR
                        )
                    
                    detections = detector.detect(inference_frame, scale_factor=1.0/scale if scale < 1.0 else 1.0)
                    last_detections = detections
                    
                    # Publish detection events
                    if detections:
                        await event_manager.publish_detection(cam_id, detections, timestamp)
                else:
                    detections = []
            
            # Annotate frame
            annotated = annotator.annotate(
                frame, 
                detections,
                fps=fps_ingest,
                cam_id=cam_id,
            )
            
            # Resize to output resolution if needed
            import cv2
            out_w, out_h = config.restream.resolution
            if annotated.shape[1] != out_w or annotated.shape[0] != out_h:
                annotated = cv2.resize(annotated, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            
            # Send to restreamer
            if cam_id in restreamers:
                await restreamers[cam_id].write_frame(annotated)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Pipeline error for {cam_id}: {e}")
            await asyncio.sleep(0.1)


# Create FastAPI app
app = FastAPI(
    title="Multi-Camera AI Service",
    description="RTSP ingest + AI detection + annotation + re-streaming",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ai-service"}


@app.get("/api/cameras")
async def get_cameras():
    """Get list of cameras and their status."""
    cameras = []
    for cam in config.cameras:
        status = await ingest_manager.get_status(cam.id)
        cameras.append({
            "id": cam.id,
            "rtsp": cam.rtsp,
            "status": status,
            "annotated_rtsp": f"{config.restream.rtsp_base}/ann_{cam.id}",
        })
    return {"cameras": cameras}


@app.get("/api/stats")
async def get_stats():
    """Get pipeline statistics."""
    stats = {
        "cameras": {},
        "ai": {
            "enabled": config.ai.enabled,
            "frame_interval": config.ai.frame_interval,
            "det_conf": config.ai.det_conf,
            "reid_enabled": config.ai.reid.enabled,
        },
    }
    
    for cam in config.cameras:
        cam_stats = await ingest_manager.get_stats(cam.id)
        restreamer_stats = restreamers[cam.id].get_stats() if cam.id in restreamers else {}
        stats["cameras"][cam.id] = {
            **cam_stats,
            "restreamer": restreamer_stats,
        }
    
    return stats


@app.get("/api/events/stream")
async def events_stream():
    """SSE endpoint for detection events."""
    from sse_starlette.sse import EventSourceResponse
    
    async def event_generator():
        async for event in event_manager.subscribe():
            yield {
                "event": "detection",
                "data": event,
            }
    
    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn
    
    cfg = get_config()
    uvicorn.run(
        "app.main:app",
        host=cfg.api.host,
        port=cfg.api.port,
        reload=False,
        log_level="info",
    )
