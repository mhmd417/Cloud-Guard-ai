from fastapi import APIRouter, Depends, HTTPException, Request, status

from backend.app.schemas.prediction import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from backend.app.security import verify_api_key
from backend.app.services.model_service import ModelNotLoadedError, ModelService

router = APIRouter()


def get_model_service(request: Request) -> ModelService:
    """Retrieve the singleton ModelService initialized during app startup."""
    service: ModelService | None = getattr(request.app.state, "model_service", None)
    if service is None or not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model service is not initialized.",
        )
    return service


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(request: Request) -> HealthResponse:
    """Liveness and readiness check verifying the ML model is loaded."""
    service: ModelService | None = getattr(request.app.state, "model_service", None)
    if service is None or not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=service.model_version,
    )


@router.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def model_info(
    service: ModelService = Depends(get_model_service),
) -> ModelInfoResponse:
    """Return public model metadata, threshold, test metrics, and disclaimers."""
    try:
        info = service.get_model_info()
    except ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return ModelInfoResponse(**info)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Prediction"],
)
def predict_failure(
    payload: PredictionRequest,
    service: ModelService = Depends(get_model_service),
) -> PredictionResponse:
    """Predict machine failure class and probability from sensor readings."""
    try:
        return service.predict(payload)
    except ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
