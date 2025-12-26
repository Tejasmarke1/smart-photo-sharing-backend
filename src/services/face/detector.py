from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DetectedFace:
    """Face detection result."""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    landmarks: Optional[np.ndarray] = None  # 5 points: eyes, nose, mouth corners
    
    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]


class RetinaFaceDetector:
    """
    InsightFace RetinaFace detector wrapper (det_10g SCRFD model).
    
    Features:
    - High-accuracy face detection using SCRFD
    - 5-point facial landmarks
    - Confidence filtering
    - Batch processing
    - CPU/GPU automatic detection
    """
    
    def __init__(
        self,
        conf_threshold: float = 0.8,
        device: str = "cpu"
    ):
        """
        Initialize RetinaFace detector using InsightFace.
        
        Args:
            conf_threshold: Confidence threshold for detections
            device: Device to use ('cpu' or 'gpu')
        """
        self.conf_threshold = conf_threshold
        self.device = device
        self.det_size = (480, 480) if device == 'cpu' else (640, 640)
        
        logger.info(f"Initializing InsightFace RetinaFace detector on {device.upper()}")
        
        try:
            from insightface.app import FaceAnalysis
            
            # Load FaceAnalysis with buffalo_l model (includes det_10g SCRFD detector)
            self.face_analysis = FaceAnalysis('buffalo_l')
            self.detector = self.face_analysis.det_model
            
            logger.info(f"✓ InsightFace RetinaFace detector initialized on {device.upper()}")
        except ImportError as e:
            logger.error(f"Failed to import InsightFace: {e}")
            raise RuntimeError(f"InsightFace is required for face detection: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize detector: {type(e).__name__}: {e}")
            raise RuntimeError(f"Failed to initialize detector: {e}")
    
    def detect(
        self, 
        image: np.ndarray,
        min_face_size: int = 20,
        max_faces: Optional[int] = None
    ) -> List[DetectedFace]:
        """
        Detect faces in image using InsightFace RetinaFace.
        
        Args:
            image: RGB image (H, W, 3)
            min_face_size: Minimum face size in pixels
            max_faces: Maximum number of faces to return
            
        Returns:
            List of DetectedFace objects, sorted by confidence
        """
        # Run detection - returns (bboxes, landmarks) tuple
        # bboxes: (N, 5) array [x1, y1, x2, y2, confidence]
        # landmarks: (N, 5, 2) array with 5 keypoints per face
        bboxes, landmarks = self.detector.detect(image, input_size=self.det_size)
        
        faces = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2, confidence = bbox
            
            # Skip low confidence detections
            if confidence < self.conf_threshold:
                continue
            
            # Convert from (x1, y1, x2, y2) to (x, y, w, h)
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            
            # Filter by size
            if w < min_face_size or h < min_face_size:
                continue
            
            # Extract landmarks if available
            face_landmarks = None
            if landmarks is not None and i < len(landmarks):
                # landmarks[i] has shape (5, 2) - 5 keypoints with x, y coordinates
                face_landmarks = landmarks[i].astype(np.float32)
            
            faces.append(DetectedFace(
                bbox=(x, y, w, h),
                confidence=float(confidence),
                landmarks=face_landmarks
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