# RetinaAI — Backend Setup Guide

Full backend connecting the website to your trained CNN model.

---

## 📁 Project Structure

```
retina-backend/
├── app.py              ← Flask API server (main entry point)
├── config.py           ← All settings (mode, paths, image size)
├── requirements.txt    ← Python dependencies
├── .env.example        ← Copy to .env and configure
├── src/
│   ├── predictor.py    ← Core inference logic (TF Serving OR Direct TF)
│   └── utils.py        ← Image validation helpers
└── uploads/            ← Temp folder (auto-created, auto-cleaned)
```

---

## ⚡ Quick Start (5 minutes)

### Step 1 — Place this backend next to your existing project

```
cv-diabetic-retinopathy-detection/
├── src/               ← your existing code
├── exported_model/    ← your exported SavedModel
├── retina-backend/    ← THIS FOLDER (place here)
└── ...
```

### Step 2 — Install dependencies

```bash
cd retina-backend
pip install -r requirements.txt
```

### Step 3 — Export your trained model (if not done yet)

```bash
# From your project root:
python src/export.py --model_dir saved_models/latest --export_dir exported_model/1
```

### Step 4 — Configure

```bash
cp .env.example .env
```

Edit `.env`:
```env
# For direct model loading (easiest):
INFERENCE_MODE=DIRECT_TF
SAVED_MODEL_PATH=../exported_model/1   # path relative to retina-backend/

# OR for TF Serving Docker:
# INFERENCE_MODE=TF_SERVING
# TF_SERVING_URL=http://localhost:8501/v1/models/retinopathy:predict
```

### Step 5 — Run the backend

```bash
python app.py
```

You should see:
```
2025-01-01 12:00:00  INFO     Loading SavedModel from: ../exported_model/1
2025-01-01 12:00:03  INFO     Model loaded successfully ✓
2025-01-01 12:00:03  INFO     Starting RetinaAI backend on port 5000 | mode=DIRECT_TF
 * Running on http://0.0.0.0:5000
```

### Step 6 — Open the website

Open `retinopathy-website.html` in your browser. The status indicator should show:
> 🟢 Backend connected

Upload a retina image and get real predictions!

---

## 🐳 Option B: TF Serving Mode (Docker)

If you prefer to use TF Serving (as in your original project):

```bash
# 1. Start TF Serving container (from your project root)
docker run -p 8501:8501 \
  --mount type=bind,source=$(pwd)/exported_model,target=/models/retinopathy \
  -e MODEL_NAME=retinopathy -t tensorflow/serving

# 2. Set inference mode in .env
INFERENCE_MODE=TF_SERVING
TF_SERVING_URL=http://localhost:8501/v1/models/retinopathy:predict

# 3. Start Flask backend
python app.py
```

---

## 🌐 API Reference

### `GET /api/health`
Check if backend and model are ready.

**Response:**
```json
{
  "status": "ok",
  "inference_mode": "DIRECT_TF",
  "model_ready": true
}
```

---

### `POST /api/predict`
Upload a retina image and get a prediction.

**Request:** `multipart/form-data` with field `image` (JPG / PNG / WEBP)

**Response:**
```json
{
  "grade": 2,
  "name": "Moderate DR",
  "confidence": 0.7134,
  "probabilities": [0.04, 0.09, 0.71, 0.12, 0.04],
  "grade_info": {
    "grade": 2,
    "name": "Moderate DR",
    "full_name": "Moderate Non-Proliferative DR",
    "description": "More than microaneurysms. Possible haemorrhages and exudates.",
    "severity": "moderate",
    "color": "#E67E22",
    "recommendation": "Ophthalmology referral within 6 months."
  },
  "mode": "DIRECT_TF"
}
```

**Error response:**
```json
{ "error": "Could not decode image. Please upload a valid JPG, PNG, or WEBP file." }
```

---

### `GET /api/grades`
Get metadata for all 5 DR grades.

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `Model not found` | Check `SAVED_MODEL_PATH` in `.env` points to your `exported_model/1` folder |
| `Cannot connect to TF Serving` | Make sure Docker container is running: `docker ps` |
| `CORS error` in browser | Backend uses flask-cors — make sure it's installed |
| `opencv error` | Run: `pip install opencv-python` |
| `503 on /api/health` | Model failed to load — check the path and TF version |

---

## 🚀 Production Deployment

For production, replace `python app.py` with gunicorn:

```bash
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

Or with Docker:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
```
