"""
utils.py — Helper utilities for the RetinaAI backend.
"""

import os
import cv2
import config


def allowed_file(filename: str) -> bool:
    """Check if a filename has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


def validate_image(path: str) -> tuple[bool, str]:
    """
    Validate that the uploaded file is a real, readable image.
    Returns (True, "") on success or (False, error_message) on failure.
    """
    if not os.path.exists(path):
        return False, "File not found after upload."

    # File size check
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > config.MAX_FILE_SIZE_MB:
        return False, f"File too large ({size_mb:.1f} MB). Maximum is {config.MAX_FILE_SIZE_MB} MB."

    # OpenCV readability check
    img = cv2.imread(path)
    if img is None:
        return False, "Could not decode image. Please upload a valid JPG, PNG, or WEBP file."

    # Minimum size check
    h, w = img.shape[:2]
    if h < 32 or w < 32:
        return False, f"Image too small ({w}×{h}px). Minimum size is 32×32px."

    return True, ""
