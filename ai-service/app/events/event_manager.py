"""
Event Manager
Manages detection events and SSE broadcasting.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, AsyncIterator
from datetime import datetime

from app.inference.detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class DetectionEvent:
    """Detection event data."""
    timestamp: str
    camera_id: str
    detections: List[Dict[str, Any]]
    count: int


class EventManager:
    """
    Manages detection events and broadcasts to SSE subscribers.
    """
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self._history: List[DetectionEvent] = []
        self._subscribers: List[asyncio.Queue] = []
        self._lock = asyncio.Lock()
    
    async def publish_detection(
        self,
        camera_id: str,
        detections: List[Detection],
        timestamp: float,
    ):
        """Publish a detection event."""
        event = DetectionEvent(
            timestamp=datetime.fromtimestamp(timestamp).isoformat(),
            camera_id=camera_id,
            detections=[
                {
                    "x1": d.x1,
                    "y1": d.y1,
                    "x2": d.x2,
                    "y2": d.y2,
                    "confidence": round(d.confidence, 2),
                    "label": d.label,
                }
                for d in detections
            ],
            count=len(detections),
        )
        
        # Add to history
        async with self._lock:
            self._history.append(event)
            if len(self._history) > self.max_history:
                self._history.pop(0)
            
            # Broadcast to subscribers
            event_json = json.dumps(asdict(event))
            for queue in self._subscribers:
                try:
                    queue.put_nowait(event_json)
                except asyncio.QueueFull:
                    pass  # Drop if subscriber is too slow
    
    async def subscribe(self) -> AsyncIterator[str]:
        """Subscribe to detection events."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        
        async with self._lock:
            self._subscribers.append(queue)
        
        try:
            # Send recent history first
            for event in self._history[-10:]:
                yield json.dumps(asdict(event))
            
            # Stream new events
            while True:
                event_json = await queue.get()
                yield event_json
        finally:
            async with self._lock:
                self._subscribers.remove(queue)
    
    def get_history(self, limit: int = 50) -> List[Dict]:
        """Get recent event history."""
        return [asdict(e) for e in self._history[-limit:]]
