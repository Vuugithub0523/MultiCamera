"""
WebSocket Handler
Streams processed frames to clients
"""
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json


class WebSocketManager:
    """Manages WebSocket connections for streaming"""
    
    def __init__(self):
        # Map of camera_id -> set of connected websockets
        self.connections: Dict[str, Set[WebSocket]] = {}
        # Latest frame for each camera
        self.latest_frames: Dict[str, bytes] = {}
        self.lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, camera_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        
        async with self.lock:
            if camera_id not in self.connections:
                self.connections[camera_id] = set()
            self.connections[camera_id].add(websocket)
        
        print(f"[WebSocket] Client connected to {camera_id} (total: {len(self.connections[camera_id])})")
    
    async def disconnect(self, websocket: WebSocket, camera_id: str):
        """Remove WebSocket connection"""
        async with self.lock:
            if camera_id in self.connections:
                self.connections[camera_id].discard(websocket)
                if len(self.connections[camera_id]) == 0:
                    del self.connections[camera_id]
        
        print(f"[WebSocket] Client disconnected from {camera_id}")
    
    async def broadcast(self, camera_id: str, data: bytes):
        """Broadcast frame to all connected clients for a camera"""
        if camera_id not in self.connections:
            return
        
        # Store latest frame
        self.latest_frames[camera_id] = data
        
        # Send to all connected clients
        disconnected = []
        for websocket in self.connections[camera_id]:
            try:
                await websocket.send_bytes(data)
            except Exception as e:
                print(f"[WebSocket] Error sending to client: {e}")
                disconnected.append(websocket)
        
        # Remove disconnected clients
        if disconnected:
            async with self.lock:
                for ws in disconnected:
                    self.connections[camera_id].discard(ws)
    
    def get_client_count(self, camera_id: str) -> int:
        """Get number of connected clients for a camera"""
        return len(self.connections.get(camera_id, set()))
    
    def get_all_client_counts(self) -> Dict[str, int]:
        """Get client counts for all cameras"""
        return {
            camera_id: len(clients)
            for camera_id, clients in self.connections.items()
        }
