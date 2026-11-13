import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from backend.app.schemas.prediction import PredictionRequest, PredictionResponse

logger = logging.getLogger("cloudguard.model_service")


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is requested before the model artifact is loaded."""


class ModelService:
    """Manages lifecycle and inference for the serialized scikit-learn pipeline."""

    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.pipeline: Any | None = None
        self.metadata: dict[str, Any] = {}

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None and bool(self.metadata)

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    @property
    def decision_threshold(self) -> float:
        return float(self.metadata.get("decision_threshold", 0.5))

    def load(self) -> None:
        """Load serialized pipeline and metadata JSON from disk once at startup."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {self.model_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Model metadata not found at: {self.metadata_path}")

        logger.info("Loading ML model artifact from %s", self.model_path.name)
        self.pipeline = joblib.load(self.model_path)

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        logger.info(
            "Model v%s loaded successfully (threshold=%.2f)",
            self.model_version,
            self.decision_threshold,
        )

    def predict(self, payload: PredictionRequest) -> PredictionResponse:
        """Run single-row inference using the loaded scikit-learn pipeline."""
        if not self.is_loaded or self.pipeline is None:
            raise ModelNotLoadedError("ML model is not loaded.")

        # Map API schema fields to the exact column names expected by ColumnTransformer
        features_df = pd.DataFrame(
            [
                {
                    "Type": payload.type,
                    "Air temperature [K]": payload.air_temperature_k,
                    "Process temperature [K]": payload.process_temperature_k,
                    "Rotational speed [rpm]": payload.rotational_speed_rpm,
                    "Torque [Nm]": payload.torque_nm,
                    "Tool wear [min]": payload.tool_wear_min,
                }
            ]
        )

        failure_prob = float(self.pipeline.predict_proba(features_df)[0, 1])
        failure_class = 1 if failure_prob >= self.decision_threshold else 0

        logger.info(
            "Prediction completed: failure_class=%d, failure_probability=%.4f, model_version=%s",
            failure_class,
            failure_prob,
            self.model_version,
        )

        return PredictionResponse(
            failure_class=failure_class,
            failure_probability=round(failure_prob, 4),
            threshold=self.decision_threshold,
            model_version=self.model_version,
        )

    def get_model_info(self) -> dict[str, Any]:
        """Return safe public metadata for GET /model-info."""
        if not self.is_loaded:
            raise ModelNotLoadedError("ML model metadata is not loaded.")

        return {
            "model_name": self.metadata.get("model_name", "CloudGuard AI Classifier"),
            "model_version": self.model_version,
            "algorithm": self.metadata.get("algorithm", "RandomForestClassifier"),
            "is_synthetic_data": bool(
                self.metadata.get("dataset", {}).get("is_synthetic", True)
            ),
            "features": self.metadata.get("features", []),
            "decision_threshold": self.decision_threshold,
            "test_metrics": self.metadata.get("test_metrics", {}),
            "disclaimer": (
                "Educational demo trained on synthetic UCI AI4I 2020 data. "
                "Does not provide advance failure warning and is not validated for industrial use."
            ),
        }
