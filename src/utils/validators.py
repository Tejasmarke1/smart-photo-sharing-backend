# src/utils/validators.py

from typing import Tuple
import numpy as np
import cv2

# ---------------------------------------------------------------------
# Configuration (tweakable)
# ---------------------------------------------------------------------

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024   # 10 MB
MAX_IMAGE_WIDTH = 8000
MAX_IMAGE_HEIGHT = 8000
MIN_IMAGE_WIDTH = 32
MIN_IMAGE_HEIGHT = 32


# ---------------------------------------------------------------------
# Custom Errors (optional but clean)
# ---------------------------------------------------------------------

class ImageValidationError(ValueError):
    """Raised when image validation fails."""


# ---------------------------------------------------------------------
# Main Validator
# ---------------------------------------------------------------------

def validate_image(
    contents: bytes,
    content_type: str,
    *,
    convert_to_rgb: bool = True
) -> np.ndarray:
    """
    Validate and decode an image safely.

    Args:
        contents: Raw image bytes
        content_type: MIME type (image/jpeg, image/png)
        convert_to_rgb: Convert BGR -> RGB (default True)

    Returns:
        np.ndarray: Decoded image (RGB if convert_to_rgb=True)

    Raises:
        ImageValidationError
    """

    # --------------------------------------------------
    # 1. MIME type check
    # --------------------------------------------------
    if content_type not in ALLOWED_MIME_TYPES:
        raise ImageValidationError(
            f"Unsupported image type: {content_type}"
        )

    # --------------------------------------------------
    # 2. File size check
    # --------------------------------------------------
    if not contents or len(contents) == 0:
        raise ImageValidationError("Empty image file")

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise ImageValidationError(
            f"Image too large (max {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB)"
        )

    # --------------------------------------------------
    # 3. Decode image safely
    # --------------------------------------------------
    np_buffer = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if image is None:
        raise ImageValidationError("Invalid or corrupted image")

    # --------------------------------------------------
    # 4. Dimension checks
    # --------------------------------------------------
    height, width = image.shape[:2]

    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        raise ImageValidationError(
            f"Image too small ({width}x{height})"
        )

    if width > MAX_IMAGE_WIDTH or height > MAX_IMAGE_HEIGHT:
        raise ImageValidationError(
            f"Image too large in dimensions ({width}x{height})"
        )

    # --------------------------------------------------
    # 5. Convert to RGB if needed
    # --------------------------------------------------
    if convert_to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image
