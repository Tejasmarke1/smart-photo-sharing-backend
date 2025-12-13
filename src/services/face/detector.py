from typing import List, Tuple, Optional
import numpy as np
import cv2
from retinaface import RetinaFace as RF
from dataclasses import dataclass

@dataclass
class DetectedFace:
    """Face detection result."""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    landmarks: np.ndarray  # 5 points: eyes, nose, mouth corners
    
    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]


class RetinaFaceDetector:
    """
    RetinaFace detector wrapper with batching and optimization.
    
    Features:
    - Multi-scale detection
    - NMS post-processing
    - Confidence filtering
    - Batch processing
    """
    
    def __init__(
        self,
        model_path: str = "models/retinaface.pth",
        conf_threshold: float = 0.8,
        nms_threshold: float = 0.4,
        device: str = "cuda:0"
    ):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        
        # Initialize RetinaFace
        self.detector = RF(
            model_path=model_path,
            device=device
        )
    
    def detect(
        self, 
        image: np.ndarray,
        min_face_size: int = 20,
        max_faces: Optional[int] = None
    ) -> List[DetectedFace]:
        """
        Detect faces in image.
        
        Args:
            image: RGB image (H, W, 3)
            min_face_size: Minimum face size in pixels
            max_faces: Maximum number of faces to return
            
        Returns:
            List of DetectedFace objects, sorted by confidence
        """
        # Run detection
        detections = self.detector.detect_faces(
            image,
            threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold
        )
        
        faces = []
        for key, detection in detections.items():
            bbox = detection['facial_area']  # [x1, y1, x2, y2]
            x, y = bbox[0], bbox[1]
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            # Filter by size
            if w < min_face_size or h < min_face_size:
                continue
            
            # Extract landmarks (left eye, right eye, nose, left mouth, right mouth)
            landmarks = np.array([
                detection['landmarks']['left_eye'],
                detection['landmarks']['right_eye'],
                detection['landmarks']['nose'],
                detection['landmarks']['mouth_left'],
                detection['landmarks']['mouth_right']
            ], dtype=np.float32)
            
            faces.append(DetectedFace(
                bbox=(x, y, w, h),
                confidence=detection['score'],
                landmarks=landmarks
            ))
        
        # Sort by confidence
        faces.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit number of faces
        if max_faces:
            faces = faces[:max_faces]
        
        return faces
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        **kwargs
    ) -> List[List[DetectedFace]]:
        """Batch detection for multiple images."""
        return [self.detect(img, **kwargs) for img in images]