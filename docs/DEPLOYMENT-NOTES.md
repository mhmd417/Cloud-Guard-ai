# CloudGuard AI — Container Deployment Notes

## 1. Container Overview
* **Base Image:** `python:3.12-slim`
* **Container User:** Non-root user (`appuser`)
* **Default Container Port:** `8000` (configurable via the `PORT` environment variable)
* **Health Check Endpoint:** `GET /health` (returns HTTP `200` when the ML model is loaded in memory, or `503` otherwise)
* **Model Artifact Location (Inside Container):**
  * Pipeline artifact: `/app/ml/models/cloudguard_rf_v1.0.0.joblib`
  * Metadata JSON: `/app/ml/models/cloudguard_rf_v1.0.0_metadata.json`
* **Model Packaging Strategy:** Both model files are baked (`COPY`) directly into the image at build time so the container is self-contained, immutable, and guaranteed to match the pinned `scikit-learn` and `joblib` versions in `backend/requirements.txt`.

---

## 2. Environment Variables

All runtime configuration is supplied through environment variables—no secrets are hardcoded in the image.

| Variable | Required | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `PORT` | No | `8000` | TCP port Uvicorn binds to inside the container (`0.0.0.0:$PORT`). |
| `API_KEY` | **Yes** | `change-me-local-dev-key` | Secret key expected in the `X-API-Key` HTTP header on `POST /predict` (minimum 8 characters). Always override in runtime environments. |
| `ALLOWED_ORIGIN` | **Yes** | `http://localhost:5500` | Single explicit frontend URL allowed by CORS (wildcard `*` is forbidden). |
| `MODEL_PATH` | No | `/app/ml/models/cloudguard_rf_v1.0.0.joblib` | Path to the serialized scikit-learn pipeline inside the container. |
| `METADATA_PATH` | No | `/app/ml/models/cloudguard_rf_v1.0.0_metadata.json` | Path to the model metadata JSON file inside the container. |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). |

---

## 3. Build & Run Instructions (Windows PowerShell)

### Build the Docker Image
Run from the repository root (`cloudguard-ai/`) so Docker can access both `backend/` and `ml/models/`:

```powershell
docker build -f backend/Dockerfile -t cloudguard-backend:1.0.0 .
```
