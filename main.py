"""
Native AI Backend Server
Multi-camera person tracking with RTSP input

Run with: python main.py
"""
import asyncio
import signal
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import Config
from core import MultiCameraManager
from api import WebSocketManager, create_rest_router


# Global instances
manager: MultiCameraManager = None
ws_manager: WebSocketManager = None
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global manager, ws_manager
    
    print("=" * 60)
    print("NATIVE AI BACKEND - Multi-Camera Tracking System")
    print("=" * 60)
    
    # Startup
    print("\n[Startup] Initializing...")
    
    # Create managers
    manager = MultiCameraManager(Config)
    ws_manager = WebSocketManager()
    
    # Define frame callback for broadcasting
    async def frame_callback(camera_id: str, jpeg_bytes: bytes, tracks):
        """Called when a frame is processed"""
        await ws_manager.broadcast(camera_id, jpeg_bytes)
    
    # Start processing in background
    processing_task = asyncio.create_task(manager.start_processing(frame_callback))
    
    print(f"\n[Startup] Server ready on http://{Config.HOST}:{Config.PORT}")
    print(f"[Startup] Cameras: {', '.join(manager.get_camera_ids())}")
    print(f"[Startup] WebSocket: ws://{Config.HOST}:{Config.PORT}/ws/tracking/{{camera_id}}")
    print("=" * 60)
    
    yield
    
    # Shutdown
    print("\n[Shutdown] Stopping...")
    manager.stop_all()
    processing_task.cancel()
    try:
        await processing_task
    except asyncio.CancelledError:
        pass
    
    print("[Shutdown] Server stopped")


# Create FastAPI app
app = FastAPI(
    title="Native AI Backend",
    description="Multi-camera person tracking with RTSP input",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# WebSocket endpoint for streaming
@app.websocket("/ws/tracking/{camera_id}")
async def websocket_tracking(websocket: WebSocket, camera_id: str):
    """
    WebSocket endpoint for receiving processed frames
    
    Usage:
        ws://localhost:5000/ws/tracking/cam01
    """
    # Check if camera exists
    if camera_id not in manager.get_camera_ids():
        await websocket.close(code=1008, reason="Camera not found")
        return
    
    await ws_manager.connect(websocket, camera_id)
    
    try:
        # Keep connection alive
        while True:
            # Wait for client messages (e.g., ping)
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Echo back for keepalive
                if message == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # No message received, just continue
                pass
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
    finally:
        await ws_manager.disconnect(websocket, camera_id)


# REST API routes
app.include_router(create_rest_router(manager, ws_manager, start_time))


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Native AI Backend",
        "version": "1.0.0",
        "cameras": manager.get_camera_ids() if manager else [],
        "endpoints": {
            "health": "/health",
            "cameras": "/api/cameras",
            "persons": "/api/persons",
            "stats": "/api/stats",
            "websocket": "/ws/tracking/{camera_id}"
        }
    }


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n[Signal] Received interrupt signal")
    raise KeyboardInterrupt


if __name__ == "__main__":
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Run server
        uvicorn.run(
            app,
            host=Config.HOST,
            port=Config.PORT,
            log_level="info",
            access_log=False  # Disable access log for cleaner output
        )
    except KeyboardInterrupt:
        print("\n[Main] Shutting down...")
