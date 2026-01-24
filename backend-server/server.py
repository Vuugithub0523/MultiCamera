import asyncio
import signal
import sys
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.stream_manager import StreamManager, load_rtsp_urls, load_tracking_config, build_camera_ids

DEFAULT_CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifecycle manager - starts and stops stream processing"""
    rtsp_urls = load_rtsp_urls(DEFAULT_CONFIG_PATH)
    camera_ids = build_camera_ids(rtsp_urls)
    tracking_config = load_tracking_config(DEFAULT_CONFIG_PATH)
    
    manager = StreamManager(rtsp_urls, camera_ids, tracking_config)
    manager.start()
    app.state.stream_manager = manager
    
    print(f"\n{'='*80}")
    print(f"Backend Server Started")
    print(f"{'='*80}")
    print(f"Cameras: {len(camera_ids)} ({', '.join(camera_ids)})")
    print(f"Tracking: {'Enabled' if tracking_config.enabled else 'Disabled'}")
    print(f"Device: {tracking_config.device.upper()}")
    print(f"{'='*80}\n")
    
    try:
        yield
    finally:
        print("\n Shutting down backend server...")
        manager.stop()


app = FastAPI(
    title="MultiCamera Backend API",
    description="Backend server for multi-camera person tracking system",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint"""
    return {
        "service": "MultiCamera Backend",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "ok"}


@app.get("/api/cameras")
async def get_cameras() -> Dict[str, list]:
    """Get list of available cameras"""
    manager: StreamManager = app.state.stream_manager
    return {
        "cameras": list(manager.loaders.keys())
    }


@app.get("/api/tracking/stats")
async def get_tracking_stats() -> Dict:
    """Get tracking statistics"""
    manager: StreamManager = app.state.stream_manager
    if manager._tracking_pipeline:
        stats = manager._tracking_pipeline.lifecycle_manager.get_statistics()
        return stats
    return {"error": "Tracking not enabled"}


@app.get("/api/tracking/events")
async def get_tracking_events(limit: int = 50) -> Dict:
    """Get recent tracking events"""
    manager: StreamManager = app.state.stream_manager
    if manager._tracking_pipeline:
        events = manager._tracking_pipeline.lifecycle_manager.get_recent_events(limit)
        return {"events": events}
    return {"error": "Tracking not enabled", "events": []}


@app.get("/api/report/stats")
async def get_report_stats() -> Dict:
    """Get report statistics including hourly traffic and camera flow"""
    manager: StreamManager = app.state.stream_manager
    if not manager._tracking_pipeline:
        return {"error": "Tracking not enabled"}
    
    lifecycle = manager._tracking_pipeline.lifecycle_manager
    stats = lifecycle.get_statistics()
    hourly = lifecycle.get_hourly_traffic()
    flow = lifecycle.get_camera_flow()
    
    # Calculate additional metrics
    all_persons = lifecycle.get_all_persons()
    total_duration = sum(p.get_duration() for p in all_persons.values())
    avg_dwell_time = total_duration / len(all_persons) if all_persons else 0
    
    # Find peak hour
    peak_hour_data = max(hourly, key=lambda x: x['count']) if hourly else {"hour": "N/A", "count": 0}
    
    return {
        "total_unique_visitors": stats['total_persons'],
        "avg_dwell_time_seconds": round(avg_dwell_time, 2),
        "peak_hour": peak_hour_data['hour'],
        "peak_hour_count": peak_hour_data['count'],
        "active_cameras": len(manager.loaders),
        "total_cameras": len(manager.loaders),
        "hourly_traffic": hourly,
        "camera_flow": flow,
        "session_duration": stats['session_duration']
    }


@app.websocket("/ws/events")
async def stream_events(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for streaming real-time tracking events
    """
    await websocket.accept()
    manager: StreamManager = websocket.app.state.stream_manager
    
    if not manager._tracking_pipeline:
        await websocket.close(code=1008, reason="Tracking not enabled")
        return
    
    try:
        last_event_count = 0
        while True:
            lifecycle = manager._tracking_pipeline.lifecycle_manager
            events = lifecycle.get_recent_events(100)
            
            # Send only new events
            if len(events) > last_event_count:
                new_events = events[last_event_count:]
                for event in new_events:
                    await websocket.send_json(event)
                last_event_count = len(events)
            
            await asyncio.sleep(0.1)  # Check for new events every 100ms
            
    except WebSocketDisconnect:
        print("  Client disconnected from events stream")
    except Exception as e:
        print(f"  Error in events WebSocket: {e}")


@app.websocket("/ws/{stream_type}/{camera_id}")
async def stream_camera(websocket: WebSocket, stream_type: str, camera_id: str) -> None:
    """
    WebSocket endpoint for streaming camera feeds
    stream_type: 'raw' or 'tracking'
    camera_id: camera identifier (e.g., 'cam01', 'cam02')
    """
    await websocket.accept()
    manager: StreamManager = websocket.app.state.stream_manager
    
    # Validate camera exists
    if manager.get_loader(camera_id) is None:
        await websocket.close(code=1008, reason=f"Camera {camera_id} not found")
        return

    try:
        while True:
            frame = manager.get_latest_frame(camera_id, stream_type)
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            
            # Encode frame as JPEG
            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                await asyncio.sleep(0.01)
                continue
            
            # Send frame to client
            await websocket.send_bytes(buffer.tobytes())
            await asyncio.sleep(0.01)  # ~100 FPS max
            
    except WebSocketDisconnect:
        print(f"  Client disconnected from {camera_id} ({stream_type})")
    except Exception as e:
        print(f"  Error in WebSocket: {e}")


def main() -> None:
    """Run the backend server"""
    import uvicorn
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: None)
    main()
