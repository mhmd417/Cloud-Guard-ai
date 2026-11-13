from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Validated machine telemetry input matching UCI AI4I 2020 features."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "type": "L",
                "air_temperature_k": 298.1,
                "process_temperature_k": 308.6,
                "rotational_speed_rpm": 1551,
                "torque_nm": 42.8,
                "tool_wear_min": 0,
            }
        },
    )

    type: Literal["L", "M", "H"] = Field(
        ...,
        description="Product quality variant: L (Low), M (Medium), or H (High).",
    )
    air_temperature_k: float = Field(
        ...,
        ge=250.0,
        le=350.0,
        description="Ambient air temperature in Kelvin (250 K to 350 K).",
    )
    process_temperature_k: float = Field(
        ...,
        ge=250.0,
        le=400.0,
        description="Machine process temperature in Kelvin (250 K to 400 K).",
    )
    rotational_speed_rpm: int = Field(
        ...,
        ge=500,
        le=5000,
        description="Spindle rotational speed in RPM (500 to 5000 rpm).",
    )
    torque_nm: float = Field(
        ...,
        ge=0.0,
        le=150.0,
        description="Torque in Newton-meters (0.0 to 150.0 Nm).",
    )
    tool_wear_min: int = Field(
        ...,
        ge=0,
        le=500,
        description="Cumulative tool wear in minutes (0 to 500 min).",
    )


class PredictionResponse(BaseModel):
    """Prediction payload returned by POST /predict."""

    failure_class: Literal[0, 1] = Field(
        ...,
        description="Predicted class: 0 = Normal operation, 1 = Predicted machine failure.",
    )
    failure_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability of machine failure (0.0 to 1.0).",
    )
    threshold: float = Field(
        ...,
        description="Decision threshold applied to convert probability into failure_class.",
    )
    model_version: str = Field(
        ...,
        description="Semantic version of the loaded model artifact.",
    )
    disclaimer: str = Field(
        default="Educational demo trained on synthetic UCI AI4I 2020 data. Not validated for real industrial use.",
        description="Mandatory safety notice.",
    )


class HealthResponse(BaseModel):
    """Service health status returned by GET /health."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    """Public model metadata returned by GET /model-info."""

    model_name: str
    model_version: str
    algorithm: str
    is_synthetic_data: bool
    features: list[str]
    decision_threshold: float
    test_metrics: dict[str, Any]
    disclaimer: str


class ErrorDetail(BaseModel):
    """Standardized error object used across all HTTP error responses."""

    code: str
    message: str
    details: Any = None


class ErrorResponse(BaseModel):
    """Top-level JSON error wrapper."""

    error: ErrorDetail
