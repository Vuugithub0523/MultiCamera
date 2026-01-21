"""
Feature Extraction for Person Re-Identification
Adapted from MultiCamera feature_extraction.py
"""
import cv2
import numpy as np
import onnxruntime as ort
from scipy.spatial import distance
from typing import Optional


class FeatureExtractor:
    """OSNet-based feature extractor for person re-identification"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize feature extractor
        
        Args:
            model_path: Path to OSNet ONNX model
            device: "cuda" or "cpu"
        """
        self.model_path = model_path
        self.device = device.lower()
        
        # Initialize ONNX Runtime session
        providers = self._get_providers()
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        
        # Get model input shape
        self.model_height, self.model_width = self.session.get_inputs()[0].shape[2:4]
        
        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        print(f"[FeatureExtractor] Loaded model: {model_path}")
        print(f"[FeatureExtractor] Input size: {self.model_width}x{self.model_height}")
        print(f"[FeatureExtractor] Device: {device}")
    
    def _get_providers(self):
        """Get ONNX Runtime execution providers"""
        if self.device == "cuda":
            # Try CUDA first, then DirectML (Windows GPU), then CPU
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "DmlExecutionProvider" in available:
                print("[FeatureExtractor] Using DirectML for GPU acceleration (Windows)")
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            else:
                print("[FeatureExtractor] WARNING: No GPU provider available, using CPU")
                providers = ["CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        return providers
    
    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract feature vector from person crop
        
        Args:
            image: Person crop (BGR format)
        
        Returns:
            Feature vector or None if image is invalid
        """
        if image is None or image.size == 0:
            return None
        
        try:
            # Preprocess
            input_tensor = self._preprocess(image)
            
            # Inference
            input_name = self.session.get_inputs()[0].name
            output = self.session.run(None, {input_name: input_tensor})
            
            # Return feature vector (flatten if needed)
            features = output[0].flatten()
            
            # Normalize (L2 normalization)
            features = features / (np.linalg.norm(features) + 1e-12)
            
            return features
            
        except Exception as e:
            print(f"[FeatureExtractor] Error extracting features: {e}")
            return None
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for feature extraction"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image_resized = cv2.resize(image_rgb, (self.model_width, self.model_height))
        
        # Convert to float32 and normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        image_normalized = (image_normalized - self.mean) / self.std
        
        # Transpose to CHW format (channels first)
        image_transposed = np.transpose(image_normalized, (2, 0, 1))
        
        # Add batch dimension: (1, C, H, W)
        tensor = np.expand_dims(image_transposed, axis=0).astype(np.float32)
        
        return tensor
    
    @staticmethod
    def compute_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compute cosine distance between two feature vectors
        
        Args:
            feat1: First feature vector
            feat2: Second feature vector
        
        Returns:
            Cosine distance (0 = identical, 1 = opposite)
        """
        return float(distance.cosine(feat1, feat2))
    
    @staticmethod
    def is_same_person(feat1: np.ndarray, feat2: np.ndarray, threshold: float = 0.42) -> bool:
        """
        Check if two feature vectors belong to the same person
        
        Args:
            feat1: First feature vector
            feat2: Second feature vector
            threshold: Distance threshold (lower = stricter)
        
        Returns:
            True if same person, False otherwise
        """
        dist = FeatureExtractor.compute_distance(feat1, feat2)
        return dist < threshold
