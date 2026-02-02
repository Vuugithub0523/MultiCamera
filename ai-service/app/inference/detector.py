"""
YOLOv4-tiny ONNX Detector
Person detection using YOLOv4-tiny with ONNXRuntime-GPU.
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Detection result."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str
    class_id: int


class YOLOv4TinyDetector:
    """
    YOLOv4-tiny detector using ONNX Runtime.
    Filters for person class only.
    """
    
    def __init__(
        self,
        onnx_path: str,
        classes_path: str,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        device: str = "cuda",
    ):
        self.onnx_path = onnx_path
        self.classes_path = classes_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = max(0.0, confidence_threshold - 0.1)
        self.device = device.lower()
        
        # Load class names
        self.class_names = self._load_classes(classes_path)
        logger.info(f"Loaded {len(self.class_names)} classes from {classes_path}")
        
        # Find person class ID
        self.person_class_id = self._find_person_class_id()
        logger.info(f"Person class ID: {self.person_class_id}")
        
        # Create ONNX session
        self.session = self._create_session()
        
        # Get model input dimensions
        input_shape = self.session.get_inputs()[0].shape
        self.model_height = input_shape[2]
        self.model_width = input_shape[3]
        logger.info(f"Model input size: {self.model_width}x{self.model_height}")
    
    def _load_classes(self, path: str) -> List[str]:
        """Load class names from file."""
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]
    
    def _find_person_class_id(self) -> int:
        """Find the class ID for 'person'."""
        for i, name in enumerate(self.class_names):
            if name.lower() == "person":
                return i
        return 0  # Default to first class
    
    def _create_session(self) -> ort.InferenceSession:
        """Create ONNX Runtime inference session."""
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        
        session = ort.InferenceSession(self.onnx_path, providers=providers)
        
        # Log which provider is being used
        active_provider = session.get_providers()[0]
        logger.info(f"ONNX Runtime using: {active_provider}")
        
        return session
    
    def detect(self, image: np.ndarray, scale_factor: float = 1.0) -> List[Detection]:
        """
        Run detection on an image.
        
        Args:
            image: BGR image
            scale_factor: Factor to scale bboxes back to original size
        
        Returns:
            List of Detection objects (person class only)
        """
        # Preprocess
        input_tensor, scale, pad = self._preprocess(image)
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        
        # Postprocess
        detections = self._postprocess(outputs, scale, pad, image.shape[:2], scale_factor)
        
        return detections
    
    def _preprocess(self, image: np.ndarray):
        """Preprocess image for YOLOv4-tiny."""
        height, width = image.shape[:2]
        
        # Calculate scale to fit model input while maintaining aspect ratio
        scale = min(self.model_width / width, self.model_height / height)
        new_w = int(width * scale)
        new_h = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.zeros((self.model_height, self.model_width, 3), dtype=np.uint8)
        pad_x = (self.model_width - new_w) // 2
        pad_y = (self.model_height - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        # Convert BGR to RGB
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Normalize and transpose to NCHW
        input_tensor = padded.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor, scale, (pad_x, pad_y)
    
    def _postprocess(
        self,
        outputs,
        scale: float,
        pad: tuple,
        original_shape: tuple,
        bbox_scale: float = 1.0,
    ) -> List[Detection]:
        """Postprocess YOLOv4-tiny outputs."""
        box_array = outputs[0]  # [batch, num_boxes, 1, 4]
        confs = outputs[1]       # [batch, num_boxes, num_classes]
        
        num_classes = confs.shape[2]
        box_array = box_array[:, :, 0]  # [batch, num_boxes, 4]
        
        # Get max confidence and class
        max_conf = np.max(confs, axis=2)  # [batch, num_boxes]
        max_id = np.argmax(confs, axis=2)  # [batch, num_boxes]
        
        detections = []
        original_h, original_w = original_shape
        pad_x, pad_y = pad
        
        # Process batch (we only have batch size 1)
        for batch_idx in range(box_array.shape[0]):
            # Filter by confidence
            mask = max_conf[batch_idx] > self.confidence_threshold
            boxes = box_array[batch_idx, mask]
            scores = max_conf[batch_idx, mask]
            class_ids = max_id[batch_idx, mask]
            
            if len(boxes) == 0:
                continue
            
            # Filter for person class only
            person_mask = class_ids == self.person_class_id
            boxes = boxes[person_mask]
            scores = scores[person_mask]
            class_ids = class_ids[person_mask]
            
            if len(boxes) == 0:
                continue
            
            # Apply NMS
            keep = self._nms(boxes, scores)
            boxes = boxes[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]
            
            # Convert to original coordinates
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                
                # Scale from model coords to original
                x1 = int((x1 * self.model_width - pad_x) / scale * bbox_scale)
                y1 = int((y1 * self.model_height - pad_y) / scale * bbox_scale)
                x2 = int((x2 * self.model_width - pad_x) / scale * bbox_scale)
                y2 = int((y2 * self.model_height - pad_y) / scale * bbox_scale)
                
                # Clip to original image bounds
                x1 = max(0, min(x1, int(original_w * bbox_scale) - 1))
                y1 = max(0, min(y1, int(original_h * bbox_scale) - 1))
                x2 = max(0, min(x2, int(original_w * bbox_scale) - 1))
                y2 = max(0, min(y2, int(original_h * bbox_scale) - 1))
                
                detections.append(Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(scores[i]),
                    label="person",
                    class_id=int(class_ids[i]),
                ))
        
        return detections
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Non-maximum suppression."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            mask = iou <= self.nms_threshold
            order = order[1:][mask]
        
        return np.array(keep)
