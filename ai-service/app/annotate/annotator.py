"""
Frame Annotator
Draw detection boxes, labels, FPS, and latency on frames.
"""

import time
from typing import List, Tuple

import cv2
import numpy as np

from app.config import AnnotateConfig
from app.inference.detector import Detection


class FrameAnnotator:
    """
    Annotates frames with detection boxes and overlays.
    """
    
    def __init__(self, config: AnnotateConfig):
        self.config = config
        self._last_frame_time = time.time()
        self._fps_history = []
        self._max_history = 30
    
    def annotate(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        fps: float = 0.0,
        cam_id: str = "",
    ) -> np.ndarray:
        """
        Annotate frame with detections and overlays.
        
        Args:
            frame: BGR image
            detections: List of Detection objects
            fps: Current FPS from ingest
            cam_id: Camera identifier
        
        Returns:
            Annotated frame (copy)
        """
        # Make a copy to avoid modifying original
        annotated = frame.copy()
        
        # Draw detection boxes
        if self.config.draw_boxes and detections:
            annotated = self._draw_detections(annotated, detections)
        
        # Calculate display FPS
        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        
        if dt > 0:
            current_fps = 1.0 / dt
            self._fps_history.append(current_fps)
            if len(self._fps_history) > self._max_history:
                self._fps_history.pop(0)
        
        avg_fps = sum(self._fps_history) / len(self._fps_history) if self._fps_history else 0
        
        # Draw overlays
        overlay_y = 30
        
        if self.config.draw_fps:
            text = f"FPS: {avg_fps:.1f} | Ingest: {fps:.1f}"
            annotated = self._draw_text(annotated, text, (10, overlay_y))
            overlay_y += 30
        
        if cam_id:
            text = f"Camera: {cam_id}"
            annotated = self._draw_text(annotated, text, (10, overlay_y))
            overlay_y += 30
        
        # Draw detection count
        text = f"Persons: {len(detections)}"
        annotated = self._draw_text(annotated, text, (10, overlay_y))
        
        return annotated
    
    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
    ) -> np.ndarray:
        """Draw detection boxes and labels."""
        box_color = self.config.box_color
        text_color = self.config.text_color
        thickness = self.config.box_thickness
        font_scale = self.config.font_scale
        
        for det in detections:
            # Draw bounding box
            cv2.rectangle(
                frame,
                (det.x1, det.y1),
                (det.x2, det.y2),
                box_color,
                thickness,
            )
            
            if self.config.draw_labels:
                # Prepare label text
                label = f"{det.label}: {det.confidence:.2f}"
                
                # Get text size
                (text_w, text_h), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    1,
                )
                
                # Draw label background
                cv2.rectangle(
                    frame,
                    (det.x1, det.y1 - text_h - 10),
                    (det.x1 + text_w + 5, det.y1),
                    box_color,
                    -1,  # Filled
                )
                
                # Draw label text
                cv2.putText(
                    frame,
                    label,
                    (det.x1 + 2, det.y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )
        
        return frame
    
    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
    ) -> np.ndarray:
        """Draw text with background."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        x, y = position
        
        # Draw background
        cv2.rectangle(
            frame,
            (x - 5, y - text_h - 5),
            (x + text_w + 5, y + 5),
            (0, 0, 0),
            -1,
        )
        
        # Draw text
        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        
        return frame
