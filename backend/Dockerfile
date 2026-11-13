# Use official slim Python 3.12 base image matching local development
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr logs
ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PORT=8000     MODEL_PATH=/app/ml/models/cloudguard_rf_v1.0.0.joblib     METADATA_PATH=/app/ml/models/cloudguard_rf_v1.0.0_metadata.json

WORKDIR /app

# Create a non-root system user and group for security
RUN groupadd --system appgroup &&     useradd --system --gid appgroup --create-home appuser

# Install pinned Python dependencies first to leverage Docker layer caching
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip &&     pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy only the backend application code and the trained model artifacts
COPY --chown=appuser:appgroup backend/app /app/backend/app
COPY --chown=appuser:appgroup ml/models/cloudguard_rf_v1.0.0.joblib /app/ml/models/cloudguard_rf_v1.0.0.joblib
COPY --chown=appuser:appgroup ml/models/cloudguard_rf_v1.0.0_metadata.json /app/ml/models/cloudguard_rf_v1.0.0_metadata.json

# Switch from root to non-root user
USER appuser

EXPOSE 8000

# Built-in container healthcheck hitting GET /health using the configured PORT
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3     CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv("PORT", "8000")}/health')" || exit 1

# Start Uvicorn listening on 0.0.0.0 and the PORT environment variable
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
