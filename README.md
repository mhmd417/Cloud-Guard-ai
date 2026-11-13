# CloudGuard AI ” Predictive Maintenance Telemetry Classifier"

[![CloudGuard AI CI](https://github.com/mhmd417/Cloud-Guard-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/mhmd417/Cloud-Guard-ai/actions/workflows/ci.yml)

An end-to-end machine learning web application that predicts industrial machine failure risk from real-time telemetry sensor readings using a calibrated **Random Forest** classifier, a **Dockerized FastAPI** inference service, and a **React + Vite + TypeScript** dashboard.

> **Educational Disclaimer:** This project is built for educational and portfolio purposes using the synthetic **UCI AI4I 2020 Predictive Maintenance Dataset**. It is **not** intended for real-world industrial safety or production control systems.

---

## System Architecture

```text
React + Vite + TypeScript (Port 5173)  ---->  FastAPI Inference Service in Docker (Port 8000)
  - Client sensor range validation             - Pydantic v2 schema validation (HTTP 422)
  - Normal / High-Risk sample presets          - X-API-Key authentication & CORS (HTTP 401)
  - Live backend status & error alerts         - Serialized scikit-learn pipeline (v1.0.0)
```

---

## Key Features

* **Leakage-Free ML Pipeline (`ml/`):** Validates the UCI AI4I 2020 schema, removes target-leaking identifier and failure-mode columns (`UDI`, tProduct ID`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`), performs stratified 80/20 train/test splits, and calibrates a decision threshold (`0.45`) optimized for failure recall and PR-AUC.
* **Production-Style FastAPI Backend (`backend/`):** Exposes `GET /health`, `GET /model-info`, and `POST /predict` with strict Pydantic range constraints, `X-API-Key` authentication, structured error envelopes (`HTTP 401`, `HTTP 422`), and automated `pytest` test suites.
* **Containerized Runtime (`backend/Dockerfile`):** Packages the inference API and serialized model artifact (`v1.0.0`) into a Docker container configurable via `PORTf¬ `API_KEY`, `ALLOWED_ORIGIN`, and `LOG_LEVEL`.
* **Interactive Telemetry Dashboard (`web/`):** Built with React, TypeScript, and Viteâ€”featuring instant sample presets, client-side range validation, probability visualization, and diagnostic controls for testing `HTTP 401` and `HTTP 422` error states.
* **Automated CI Pipeline (`.github/workflows/ci.yml`)Æª" Runs `pytest` for backend and ML plus `npm ci` & `npm run build` on every push and pull request.

---

## Project Structure

```text
cloudguard-ai/
â¦œ .github/workflows/
   â””â€” ci.yml              # GitHub Actions CI workflow (pytest + Vite build)
â’œ backend/
   â˜œ app/                 # FastAPI application, schemas, auth, and model loader
   â”œ tests/               # Pytest endpoint, auth, and validation suites
   â”œ Dockerfile            # Container build specification
   â˜” requirements.txt      # Backend Python dependencies
â’œ data/
   â˜” README.md            # Dataset acquisition & schema documentation
â”œ docs/
   â”œ DEPLOYMENT-TEST..md -> DEPLOYMENT-NOTES.md  # Docker & cloud deployment guide
   â˜” MODEL_CARD.md        # Model metrics, limitations, and ethical scope
â”œ ml/                     # Data preprocessing, training, evaluation, and unit tests
â”œ web/
   â”œ src/                 # React + TypeScript components, API client, and validation
   â””â€” package.json          # Frontend dependencies and Vite scripts
b”” README.md                # Project overview and quick-start guide
```

---

## Quick Start (Windows PowerShell)

### 1. Configure Frontend Environment
```powershell
Copy-Item -Path "web\.env.example" -Destination "web\.env" -Force
```

Ensure `web/.env` contains:
```env
VITE_API_URL=http://127.0.0.1:8000
VITE_API_KEY=cloudguard-local-dev-key-2026
```

### 2. Build & Start the Backend Container
```powershell
docker build -t cloudguard-backend:1.0.0 -f backend/Dockerfile .

docker run -d `
  --name cloudguard-api `
  -p 8000:8000 `
  -e PORT=8000 `
  -e API_KEY="cloudguard-local-dev-key-2026" `
  -e ALLOWED_ORIGIN="http://localhost:5173" `
  -e LOG_LEVEL="INFO" `
  cloudguard-backend:1.0.0

Start-Sleep -Seconds 3
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
```

### 3. Start the React + Vite Frontend
```powershell
Set-Location -Path "web"
npm install
npm run dev
```

Open **`http://localhost:5173`** in your browser.

---

## API Reference

| Method | Endpoint | Auth Required | Description |
| :--- | :--- | :---: | :--- |
| `GET` | `/health` | No | Returns service status, `model_loaded` flag, and active `model_version`. |
| `GET` | `/model-info` | No | Returns algorithm name, feature list, threshold, and test metrics. |
| `POST` | `/predict` | `X-API-Key` | Predicts binary machine failure (`0` or `1`) and failure probability. |

### Sample `POST /predict` Request
```json
{
  "type": "L",
  "air_temperature_k": 304.0,
  "process_temperature_k": 313.2,
  "rotational_speed_rpm": 1350,
  "torque_nm": 68.5,
  "tool_wear_min": 220
}
```

### Sample `POST /predict` Response (`200 OK`)
```json
{
  "failure_class": 1,
  "failure_probability": 0.94,
  "threshold": 0.45,
  "model_version": "1.0.0",
  "disclaimer": "Educational demo prediction on synthetic AI4I 2020 data; not for industrial control."
}
```

---

## Author

**Mohamed Reda Farag**
  AI & Computer Science Student â€” Sphinx University
  Email: `mhmdrdafraj100@gmail.com`
  Gemini: https://github.com/mhmd417
