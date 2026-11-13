import logging
import secrets

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from backend.app.config import Settings, get_settings

logger = logging.getLogger("cloudguard.security")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(
    provided_key: str | None = Security(api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    """Validate the X-API-Key header against the configured environment secret."""
    if not provided_key or not secrets.compare_digest(provided_key, settings.api_key):
        # Never log the provided or expected key value
        logger.warning("Rejected request with missing or invalid X-API-Key header.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
        )
