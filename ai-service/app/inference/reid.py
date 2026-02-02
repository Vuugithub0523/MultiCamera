"""
ReID Feature Extractor
Person re-identification using OSNet ONNX model.
"""

import logging
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ReIDExtractor:
    """
    Feature extractor for person re-identification.
    Uses OSNet ONNX model.
    """
    
    def __init__(
        self,
        onnx_path: str,
        device: str = "cuda",
    ):
        self.onnx_path = onnx_path
        self.device = device.lower()
        
        # Create ONNX session
        self.session = self._create_session()
        
        # Get model input dimensions
        input_shape = self.session.get_inputs()[0].shape
        self.model_height = input_shape[2]
        self.model_width = input_shape[3]
        logger.info(f"ReID model input size: {self.model_width}x{self.model_height}")
    
    def _create_session(self) -> ort.InferenceSession:
        """Create ONNX Runtime inference session."""
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        
        session = ort.InferenceSession(self.onnx_path, providers=providers)
        
        active_provider = session.get_providers()[0]
        logger.info(f"ReID ONNX Runtime using: {active_provider}")
        
        return session
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from a person crop.
        
        Args:
            image: BGR image of person crop
        
        Returns:
            Feature vector (normalized)
        """
        # Preprocess
        input_tensor = self._preprocess(image)
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        
        # Normalize feature vector
        feature = outputs[0].flatten()
        feature = feature / np.linalg.norm(feature)
        
        return feature
    
    def extract_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Extract features from multiple person crops."""
        return [self.extract(img) for img in images]
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for OSNet."""
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image = cv2.resize(image, (self.model_width, self.model_height))
        
        # Normalize (ImageNet mean/std)
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # Transpose to NCHW
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        return image
    
    @staticmethod
    def cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Calculate cosine similarity between two feature vectors."""
        return float(np.dot(feat1, feat2))
    
    @staticmethod
    def cosine_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Calculate cosine distance (1 - similarity)."""
        return 1.0 - ReIDExtractor.cosine_similarity(feat1, feat2)
