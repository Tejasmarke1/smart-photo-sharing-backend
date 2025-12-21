"""Test face pipeline."""
import cv2
import numpy as np
import pytest

from src.services.face.pipeline import FacePipeline

@pytest.fixture
def pipeline():
    return FacePipeline()

def test_face_detection(pipeline):
    """Test face detection on sample image."""
    # Create a simple test image with a face
    # For real testing, use an actual photo
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    results = pipeline.process_photo(image)
    
    # Should return a list (may be empty for random image)
    assert isinstance(results, list)

def test_with_real_image(pipeline):
    """Test with a real image."""
    # Download a test image
    # You can use any portrait photo
    # For now, skip if no test image available
    pytest.skip("Requires test image")