"""
config.py — Central configuration for RetinaAI backend.

Edit this file OR set environment variables to override.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE MODE
# ─────────────────────────────────────────────────────────────────────────────
INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "DIRECT_TF")

# ─────────────────────────────────────────────────────────────────────────────
# TF SERVING SETTINGS  (only used when INFERENCE_MODE = "TF_SERVING")
# ─────────────────────────────────────────────────────────────────────────────
TF_SERVING_URL = os.environ.get(
    "TF_SERVING_URL",
    "http://localhost:8501/v1/models/retinopathy:predict"
)
TF_SERVING_TIMEOUT = int(os.environ.get("TF_SERVING_TIMEOUT", "10"))

# ─────────────────────────────────────────────────────────────────────────────
# DIRECT TF SETTINGS  (only used when INFERENCE_MODE = "DIRECT_TF")
# ─────────────────────────────────────────────────────────────────────────────
# Uses absolute path so it works regardless of where you run app.py from.
# Falls back to env variable if set, otherwise uses the hardcoded absolute path.
_DEFAULT_MODEL_PATH = r"C:\Users\Niraj Bhirud\OneDrive\Desktop\Rtina-eye\cv-diabetic-retinopathy-detection\exported_model\1"

SAVED_MODEL_PATH = os.environ.get("SAVED_MODEL_PATH", _DEFAULT_MODEL_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# Must match what your model was trained on (src/preprocess.py in your repo)
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)   # (width, height) — Kaggle DR dataset is 224x224
IMAGE_CHANNELS = 3        # RGB
NORMALIZE = True          # divide pixels by 255.0

# ─────────────────────────────────────────────────────────────────────────────
# CLASS LABELS
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR",
]
NUM_CLASSES = len(CLASS_NAMES)  # 5

# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_FILE_SIZE_MB = 16