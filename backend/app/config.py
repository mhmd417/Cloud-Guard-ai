from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve repository root: backend/app/config.py -> parents[2] is cloudguard-ai/
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application configuration loaded strictly from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_path: Path = Field(
        default=REPO_ROOT / "ml" / "models" / "cloudguard_rf_v1.0.0.joblib",
        description="Filesystem path to the serialized scikit-learn pipeline.",
    )
    metadata_path: Path = Field(
        default=REPO_ROOT / "ml" / "models" / "cloudguard_rf_v1.0.0_metadata.json",
        description="Filesystem path to the model metadata JSON file.",
    )
    api_key: str = Field(
        default="change-me-local-dev-key",
        min_length=8,
        description="API key required in X-API-Key header for POST /predict.",
    )
    allowed_origin: str = Field(
        default="http://localhost:5500",
        description="Single explicit CORS origin allowed to call the API (never '*').",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Python logging level.",
    )

    @field_validator("allowed_origin")
    @classmethod
    def forbid_wildcard_origin(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or cleaned == "*":
            raise ValueError("ALLOWED_ORIGIN must be an explicit origin URL and cannot be '*'.")
        return cleaned


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance so env vars are parsed once."""
    return Settings()
