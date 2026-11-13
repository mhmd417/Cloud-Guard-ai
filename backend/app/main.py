from contextlib import asynccontextmanager
import logging
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import get_settings
from backend.app.routes.prediction import router as prediction_router
from backend.app.schemas.prediction import ErrorDetail, ErrorResponse
from backend.app.services.model_service import ModelService


def configure_logging(level_name: str) -> None:
    """Configure application-wide structured console logging without sensitive data."""
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the ML pipeline once at startup and release cleanly on shutdown."""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger("cloudguard.main")

    service = ModelService(
        model_path=settings.model_path,
        metadata_path=settings.metadata_path,
    )
    service.load()
    app.state.model_service = service
    logger.info("CloudGuard AI backend startup complete.")
    yield
    logger.info("CloudGuard AI backend shutting down.")


def create_app() -> FastAPI:
    """Application factory creating the FastAPI instance, CORS, and error handlers."""
    settings = get_settings()

    app = FastAPI(
        title="CloudGuard AI Predictive Maintenance API",
        version="1.0.0",
        description=(
            "Educational predictive maintenance API powered by a scikit-learn pipeline "
            "trained on the synthetic UCI AI4I 2020 dataset. Not validated for industrial use."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.allowed_origin],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        sanitized_errors = [
            {
                "field": ".".join(str(loc) for loc in err.get("loc", [])),
                "message": err.get("msg", "Invalid value"),
                "type": err.get("type", "value_error"),
            }
            for err in exc.errors()
        ]
        body = ErrorResponse(
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message="Request input failed validation.",
                details=sanitized_errors,
            )
        )
        return JSONResponse(
            status_code=422,
            content=body.model_dump(),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        code_map = {
            401: "UNAUTHORIZED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            503: "SERVICE_UNAVAILABLE",
        }
        error_code = code_map.get(exc.status_code, f"HTTP_{exc.status_code}")
        body = ErrorResponse(
            error=ErrorDetail(
                code=error_code,
                message=str(exc.detail),
                details=None,
            )
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=body.model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logging.getLogger("cloudguard.main").exception(
            "Unhandled server error on %s %s", request.method, request.url.path
        )
        body = ErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected internal error occurred.",
                details=None,
            )
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=body.model_dump(),
        )

    app.include_router(prediction_router)
    return app


app = create_app()
