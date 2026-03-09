"""
RetinaAI Backend — Flask API Server
Connects the website frontend to TF Serving (or direct TF model loading).

Two inference modes:
  1. TF_SERVING  — calls your running TF Serving Docker container
  2. DIRECT_TF   — loads saved_model directly (no Docker needed, easier)

Routes:
  GET  /                → serves static/demo.html  (your website)
  GET  /api/health      → model + server status
  POST /api/predict     → upload image, get DR grade prediction
  GET  /api/grades      → metadata for all 5 DR grades
"""

import os
import uuid
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from src.predictor import Predictor
from src.utils import allowed_file, validate_image

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)  # allow all origins — fine for local dev

app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ── Predictor singleton ───────────────────────────────────────────────────────
predictor = Predictor()


# ── Website route ─────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """
    Serve the main website.
    Looks for HTML files in static/ in this order:
      demo.html → index.html → first .html file found
    Open http://localhost:5000 in your browser.
    """
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

    # Check preferred filenames in order
    for filename in ["demo.html", "index.html"]:
        if os.path.exists(os.path.join(static_dir, filename)):
            log.info(f"Serving → static/{filename}")
            return send_from_directory("static", filename)

    # Fall back to first .html file found
    if os.path.exists(static_dir):
        html_files = [f for f in os.listdir(static_dir) if f.endswith(".html")]
        if html_files:
            log.info(f"Serving → static/{html_files[0]}")
            return send_from_directory("static", html_files[0])

    # Nothing found — helpful error
    return jsonify({
        "error": "No HTML file found in static/ folder.",
        "fix": "Place your website HTML file inside retina-backend/static/ and name it demo.html"
    }), 404


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    """Health check — confirms model / server readiness."""
    ready = predictor.health_check()
    return jsonify({
        "status": "ok",
        "inference_mode": predictor.mode,
        "model_ready": ready,
    }), 200 if ready else 503


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST /api/predict
    Body:    multipart/form-data  →  field 'image'  (JPEG / PNG / WEBP)
    Returns: JSON with grade, confidence, probabilities, grade_info

    Example response:
    {
      "grade": 2,
      "name": "Moderate DR",
      "confidence": 0.7134,
      "probabilities": [0.04, 0.09, 0.71, 0.12, 0.04],
      "grade_info": { ... },
      "mode": "DIRECT_TF"
    }
    """
    # 1. Validate request has image field
    if "image" not in request.files:
        return jsonify({"error": "No 'image' field in request. Send multipart/form-data with field named 'image'."}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Use JPG, PNG, or WEBP."}), 400

    # 2. Save to temp file
    ext = os.path.splitext(secure_filename(file.filename))[1].lower() or ".jpg"
    tmp_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4().hex}{ext}")

    try:
        file.save(tmp_path)
        log.info(f"Upload saved → {tmp_path}")

        # 3. Validate image is actually readable
        valid, err = validate_image(tmp_path)
        if not valid:
            return jsonify({"error": err}), 400

        # 4. Run inference
        result = predictor.predict(tmp_path)
        log.info(f"Result: grade={result['grade']} ({result['name']})  conf={result['confidence']:.1%}")
        return jsonify(result), 200

    except ConnectionError as e:
        log.error(str(e))
        return jsonify({"error": str(e)}), 503

    except Exception as e:
        log.exception("Prediction failed")
        return jsonify({"error": f"Inference error: {str(e)}"}), 500

    finally:
        # Always clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/api/grades", methods=["GET"])
def grades():
    """Return metadata for all 5 DR grades."""
    return jsonify({"grades": predictor.GRADE_INFO}), 200


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    log.info("─────────────────────────────────────────────────")
    log.info(f"  RetinaAI backend  →  http://localhost:{port}")
    log.info(f"  Inference mode    →  {predictor.mode}")
    log.info(f"  Website           →  http://localhost:{port}/")
    log.info("─────────────────────────────────────────────────")

    app.run(host="0.0.0.0", port=port, debug=debug)