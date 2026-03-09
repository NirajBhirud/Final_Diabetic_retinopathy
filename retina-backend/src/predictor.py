"""
predictor.py — Core inference engine.

Supports two modes:
  - TF_SERVING : sends HTTP request to your Docker TF Serving endpoint
  - DIRECT_TF  : loads the SavedModel directly using tf.saved_model.load()
"""

import logging
import numpy as np
import requests
import config

log = logging.getLogger(__name__)


class Predictor:
    """
    Unified predictor that works with either:
      1. TF Serving (Docker container) — production recommended
      2. Direct TF SavedModel loading  — easier for development
    """

    # Grade metadata used by both /api/predict and /api/grades
    GRADE_INFO = [
        {
            "grade": 0,
            "name": "No DR",
            "full_name": "No Diabetic Retinopathy",
            "description": "No signs of diabetic retinopathy. Healthy retinal vasculature.",
            "severity": "none",
            "color": "#27AE60",
            "recommendation": "Routine annual screening recommended.",
        },
        {
            "grade": 1,
            "name": "Mild DR",
            "full_name": "Mild Diabetic Retinopathy",
            "description": "Microaneurysms only. Earliest clinical sign of DR.",
            "severity": "mild",
            "color": "#2980B9",
            "recommendation": "Follow-up in 12 months. Improve glycaemic control.",
        },
        {
            "grade": 2,
            "name": "Moderate DR",
            "full_name": "Moderate Non-Proliferative DR",
            "description": "More than microaneurysms. Possible haemorrhages and exudates.",
            "severity": "moderate",
            "color": "#E67E22",
            "recommendation": "Ophthalmology referral within 6 months.",
        },
        {
            "grade": 3,
            "name": "Severe DR",
            "full_name": "Severe Non-Proliferative DR",
            "description": "Significant vessel damage in 4 quadrants. High progression risk.",
            "severity": "severe",
            "color": "#E74C3C",
            "recommendation": "Urgent ophthalmology referral within 1 month.",
        },
        {
            "grade": 4,
            "name": "Proliferative DR",
            "full_name": "Proliferative Diabetic Retinopathy",
            "description": "Neovascularisation present. Highest severity — risk of blindness.",
            "severity": "critical",
            "color": "#C0392B",
            "recommendation": "URGENT: Same-week ophthalmology assessment required.",
        },
    ]

    def __init__(self):
        self.mode = config.INFERENCE_MODE
        self._model = None  # only used in DIRECT_TF mode

        if self.mode == "DIRECT_TF":
            self._load_model()
        elif self.mode == "TF_SERVING":
            log.info(f"TF Serving mode → endpoint: {config.TF_SERVING_URL}")
        else:
            raise ValueError(f"Unknown INFERENCE_MODE: {self.mode}. Use 'TF_SERVING' or 'DIRECT_TF'.")

    # ── Model Loading (DIRECT_TF only) ────────────────────────────────────────

    def _load_model(self):
        """Load SavedModel from disk into memory."""
        import tensorflow as tf  # lazy import — not needed for TF_SERVING mode

        model_path = config.SAVED_MODEL_PATH
        log.info(f"Loading SavedModel from: {model_path}")
        try:
            self._model = tf.saved_model.load(model_path)
            # Get the default serving function
            self._infer_fn = self._model.signatures["serving_default"]
            log.info("Model loaded successfully ✓")
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            log.warning("Predictor will run in DEMO mode (random predictions).")
            self._model = None

    # ── Image Preprocessing ───────────────────────────────────────────────────

    def _preprocess(self, image_path: str) -> np.ndarray:
        """
        Preprocess a retina image to match the training pipeline.

        Steps (matching src/preprocess.py in your repo):
          1. Load image with OpenCV
          2. Convert BGR → RGB
          3. Resize to IMAGE_SIZE
          4. Normalize to [0, 1]
          5. Add batch dimension: (1, H, W, 3)
        """
        import cv2

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # BGR → RGB (OpenCV loads as BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        img = cv2.resize(img, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        # Normalize
        if config.NORMALIZE:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)

        # Add batch dimension → (1, 224, 224, 3)
        img = np.expand_dims(img, axis=0)
        return img

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, image_path: str) -> dict:
        """
        Run inference on a retina image.

        Returns:
            {
              "grade":       int,        # 0–4
              "name":        str,        # e.g. "Moderate DR"
              "confidence":  float,      # 0.0–1.0 for predicted class
              "probabilities": [float],  # list of 5 class probabilities
              "grade_info":  dict,       # full grade metadata
              "mode":        str,        # "TF_SERVING" | "DIRECT_TF" | "DEMO"
            }
        """
        img_array = self._preprocess(image_path)

        if self.mode == "TF_SERVING":
            probs = self._predict_tf_serving(img_array)
        elif self.mode == "DIRECT_TF" and self._model is not None:
            probs = self._predict_direct(img_array)
        else:
            # Demo / fallback mode when model file isn't available
            log.warning("Model not loaded — returning demo prediction.")
            probs = self._demo_prediction()

        grade = int(np.argmax(probs))
        confidence = float(probs[grade])

        return {
            "grade": grade,
            "name": config.CLASS_NAMES[grade],
            "confidence": round(confidence, 4),
            "probabilities": [round(float(p), 4) for p in probs],
            "grade_info": self.GRADE_INFO[grade],
            "mode": self.mode,
        }

    def _predict_tf_serving(self, img_array: np.ndarray) -> np.ndarray:
        """Call TF Serving REST endpoint."""
        payload = {"instances": img_array.tolist()}
        try:
            response = requests.post(
                config.TF_SERVING_URL,
                json=payload,
                timeout=config.TF_SERVING_TIMEOUT,
            )
            response.raise_for_status()
            predictions = response.json()["predictions"][0]
            probs = np.array(predictions)
            # Apply softmax if raw logits are returned
            if probs.max() > 1.0 or probs.min() < 0.0:
                probs = self._softmax(probs)
            return probs
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Cannot connect to TF Serving. "
                "Make sure Docker is running: "
                "docker run -p 8501:8501 --mount type=bind,source=$(pwd)/exported_model,"
                "target=/models/retinopathy -e MODEL_NAME=retinopathy tensorflow/serving"
            )

    def _predict_direct(self, img_array: np.ndarray) -> np.ndarray:
        """Run inference directly via loaded SavedModel."""
        import tensorflow as tf

        tensor = tf.constant(img_array, dtype=tf.float32)
        output = self._infer_fn(tensor)

        # Get the output tensor — name may vary, try common keys
        output_keys = list(output.keys())
        log.debug(f"Model output keys: {output_keys}")

        raw = output[output_keys[0]].numpy()[0]

        # Softmax if raw logits
        if raw.max() > 1.0 or raw.min() < 0.0:
            raw = self._softmax(raw)
        return raw

    def _demo_prediction(self) -> np.ndarray:
        """Return a random valid probability distribution (demo mode)."""
        raw = np.random.dirichlet(np.ones(config.NUM_CLASSES))
        return raw

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    # ── Health Check ──────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Returns True if the model/serving endpoint is ready."""
        if self.mode == "TF_SERVING":
            try:
                url = config.TF_SERVING_URL.replace(":predict", "")
                r = requests.get(url, timeout=3)
                return r.status_code == 200
            except Exception:
                return False
        elif self.mode == "DIRECT_TF":
            return self._model is not None
        return False
