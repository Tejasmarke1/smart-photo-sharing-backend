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
        landmarks: np.ndarray = None,
        bbox: Tuple[int, int, int, int] = None
    ) -> np.ndarray:
        """
        Align face using similarity transform or bbox crop fallback.
        
        Args:
            image: RGB image
            landmarks: 5 facial landmarks (5, 2), optional
            bbox: Bounding box (x, y, w, h) for fallback
            
        Returns:
            Aligned face crop (output_size)
        """
        # Try landmark-based alignment first when available and reliable
        try:
            if landmarks is not None and len(landmarks) == 5:
                # Validate landmark coordinates are finite and within image bounds
                if (
                    np.isfinite(landmarks).all()
                    and (landmarks[:, 0] >= 0).all()
                    and (landmarks[:, 1] >= 0).all()
                    and (landmarks[:, 0] < image.shape[1]).all()
                    and (landmarks[:, 1] < image.shape[0]).all()
                ):
                    # Estimate similarity transform
                    tform = cv2.estimateAffinePartial2D(
                        landmarks,
                        self.template,
                        method=cv2.LMEDS
                    )[0]
                    # If transform was found, warp image
                    if tform is not None:
                        aligned = cv2.warpAffine(
                            image,
                            tform,
                            self.output_size,
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0
                        )
                        return aligned
                # If landmarks failed validation or transform, fall through to bbox
        except Exception:
            # Any failure in landmark alignment should fallback to bbox
            pass
        # Fallback: bbox-based crop and resize
        if bbox is not None:
            # Fallback: crop and resize with padding
            x, y, w, h = bbox
            pad_w, pad_h = int(w * 0.15), int(h * 0.15)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(image.shape[1], x + w + pad_w)
            y2 = min(image.shape[0], y + h + pad_h)
            crop = image[y1:y2, x1:x2]
            aligned = cv2.resize(crop, self.output_size, interpolation=cv2.INTER_LINEAR)
            return aligned
        
        # If neither reliable landmarks nor bbox available, raise
        raise ValueError("Either reliable landmarks or bbox must be provided")
    
    def align_batch(
        self,
        image: np.ndarray,
        faces: List[DetectedFace]
    ) -> List[np.ndarray]:
        """Align multiple faces from same image."""
        return [self.align(image, face.landmarks, face.bbox) for face in faces]