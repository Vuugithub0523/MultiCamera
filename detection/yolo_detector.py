"""
YOLO Detector for person detection
Adapted from MultiCamera object_detection.py
"""
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple


class YOLODetector:
    """YOLO object detector using ONNX Runtime"""
    
    def __init__(
        self,
        model_path: str,
        coco_names_path: str,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        target_classes: List[str] = None
    ):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to ONNX model file
            coco_names_path: Path to COCO class names file
            device: "cuda" or "cpu"
            confidence_threshold: Detection confidence threshold
            target_classes: List of classes to detect (default: ["person"])
        """
        self.model_path = model_path
        self.coco_names_path = coco_names_path
        self.confidence_threshold = confidence_threshold
        self.target_classes = np.array(target_classes or ["person"])
        self.device = device.lower()
        
        # NMS threshold
        self.nms_threshold = max(0, self.confidence_threshold - 0.1)
        
        # Load class names
        with open(self.coco_names_path, "r") as f:
            self.class_names = np.array([cls.strip() for cls in f.readlines()])
        
        # Initialize ONNX Runtime session
        providers = self._get_providers()
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        
        # Get model input shape
        self.model_height, self.model_width = self.session.get_inputs()[0].shape[2:4]
        
        print(f"[YOLODetector] Loaded model: {model_path}")
        print(f"[YOLODetector] Input size: {self.model_width}x{self.model_height}")
        print(f"[YOLODetector] Device: {device}")
        print(f"[YOLODetector] Providers: {self.session.get_providers()}")
    
    def _get_providers(self):
        """Get ONNX Runtime execution providers based on device"""
        if self.device == "cuda":
            # Try CUDA first, then DirectML (Windows GPU), then CPU
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "DmlExecutionProvider" in available:
                print("[YOLODetector] Using DirectML for GPU acceleration (Windows)")
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            else:
                print("[YOLODetector] WARNING: No GPU provider available, using CPU")
                providers = ["CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        return providers
    
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect objects in frame
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            Array of detections in format [x1, y1, x2, y2, confidence]
        """
        # Preprocess
        input_tensor = self._preprocess(frame)
        
        # Inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        
        # Postprocess
        detections = self._postprocess(outputs, frame.shape[:2])
        
        # Debug: Print detection info
        if len(detections) > 0:
            print(f"[YOLODetector] Raw detections: {len(detections)}, confidences: {[d[4] for d in detections[:3]]}")
        
        return detections
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for YOLO"""
        # Resize to model input size
        image = cv2.resize(
            frame,
            (self.model_width, self.model_height),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transpose to CHW format
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        # Normalize to [0, 1]
        image /= 255.0
        
        return image
    
    def _postprocess(
        self,
        outputs: List[np.ndarray],
        original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Postprocess YOLO outputs
        
        Args:
            outputs: Raw YOLO outputs
            original_shape: Original frame shape (H, W)
        
        Returns:
            Array of detections [x1, y1, x2, y2, confidence]
        """
        box_array = outputs[0]
        confs = outputs[1]
        
        # Get dimensions
        num_classes = confs.shape[2]
        box_array = box_array[:, :, 0]
        
        # Get max confidence and class per detection
        max_conf = np.max(confs, axis=2)
        max_id = np.argmax(confs, axis=2)
        
        # Process batch (should be 1)
        all_detections = []
        for i in range(box_array.shape[0]):
            # Filter by confidence
            mask = max_conf[i] > self.confidence_threshold
            boxes = box_array[i, mask, :]
            confidences = max_conf[i, mask]
            class_ids = max_id[i, mask]
            
            # Apply NMS per class
            batch_detections = []
            for j in range(num_classes):
                class_mask = class_ids == j
                class_boxes = boxes[class_mask, :]
                class_confs = confidences[class_mask]
                class_ids_filtered = class_ids[class_mask]
                
                if len(class_boxes) == 0:
                    continue
                
                # NMS
                keep = self._nms(class_boxes, class_confs, self.nms_threshold)
                
                if len(keep) > 0:
                    nms_boxes = class_boxes[keep, :]
                    nms_confs = class_confs[keep]
                    nms_class_ids = class_ids_filtered[keep]
                    
                    for k in range(len(nms_boxes)):
                        batch_detections.append([
                            nms_boxes[k, 0], nms_boxes[k, 1],
                            nms_boxes[k, 2], nms_boxes[k, 3],
                            nms_confs[k],
                            nms_class_ids[k]
                        ])
            
            all_detections.extend(batch_detections)
        
        # Convert to numpy array
        if len(all_detections) == 0:
            return np.empty((0, 5))
        
        detections = np.array(all_detections)
        
        # Filter by target classes and convert to absolute coordinates
        filtered = []
        orig_h, orig_w = original_shape
        
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            class_name = self.class_names[int(cls_id)]
            
            # Only keep target classes
            if class_name not in self.target_classes:
                continue
            
            # Convert from normalized [0,1] to pixel coordinates
            x1 = int(x1 * orig_w)
            y1 = int(y1 * orig_h)
            x2 = int(x2 * orig_w)
            y2 = int(y2 * orig_h)
            
            # Clip to frame bounds
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))
            
            filtered.append([x1, y1, x2, y2, conf])
        
        return np.array(filtered) if filtered else np.empty((0, 5))
    
    def _nms(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        threshold: float,
        min_mode: bool = False
    ) -> np.ndarray:
        """
        Non-Maximum Suppression
        
        Args:
            boxes: Bounding boxes in format [x1, y1, x2, y2]
            confidences: Confidence scores
            threshold: IoU threshold
            min_mode: Use minimum area for IoU calculation
        
        Returns:
            Indices of boxes to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = confidences.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            idx_self = order[0]
            idx_other = order[1:]
            
            keep.append(idx_self)
            
            # Calculate IoU
            xx1 = np.maximum(x1[idx_self], x1[idx_other])
            yy1 = np.maximum(y1[idx_self], y1[idx_other])
            xx2 = np.minimum(x2[idx_self], x2[idx_other])
            yy2 = np.minimum(y2[idx_self], y2[idx_other])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            if min_mode:
                over = inter / np.minimum(areas[idx_self], areas[idx_other])
            else:
                over = inter / (areas[idx_self] + areas[idx_other] - inter)
            
            # Keep boxes with IoU less than threshold
            inds = np.where(over <= threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep, dtype=np.int32)
