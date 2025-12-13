import cv2
import numpy as np
from typing import List, Tuple
from src.services.face.detector import DetectedFace


class FaceAligner:
    """
    Align faces using landmarks for optimal embedding extraction.
    
    Based on standard face alignment to canonical pose:
    - Eyes horizontal
    - Nose centered
    - Consistent scale
    """
    
    def __init__(self, output_size: Tuple[int, int] = (112, 112)):
        self.output_size = output_size
        
        # Standard template for 112x112 face
        self.template = np.array([
            [38.2946, 51.6963],  # left eye
            [73.5318, 51.5014],  # right eye
            [56.0252, 71.7366],  # nose
            [41.5493, 92.3655],  # left mouth
            [70.7299, 92.2041]   # right mouth
        ], dtype=np.float32)
    
    def align(
        self,
        image: np.ndarray,
        landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Align face using similarity transform.
        
        Args:
            image: RGB image
            landmarks: 5 facial landmarks (5, 2)
            
        Returns:
            Aligned face crop (output_size)
        """
        # Estimate similarity transform
        tform = cv2.estimateAffinePartial2D(
            landmarks,
            self.template,
            method=cv2.LMEDS
        )[0]
        
        # Warp image
        aligned = cv2.warpAffine(
            image,
            tform,
            self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return aligned
    
    def align_batch(
        self,
        image: np.ndarray,
        faces: List[DetectedFace]
    ) -> List[np.ndarray]:
        """Align multiple faces from same image."""
        return [self.align(image, face.landmarks) for face in faces]