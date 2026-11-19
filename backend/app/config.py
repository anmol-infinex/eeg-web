"""
backend/app/config.py
Configuration settings loaded from environment variables.
"""
import os
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field

class Settings(BaseModel):
    # Security
    ACCESS_KEY: str = Field(
        default_factory=lambda: os.getenv("ACCESS_KEY", "sentry-dev-key-change-in-prod")
    )
    ALLOWED_ORIGIN: str = Field(
        default_factory=lambda: os.getenv("ALLOWED_ORIGIN", os.getenv("CORS_ORIGINS", "*"))
    )
    CORS_ORIGINS: str = Field(
        default_factory=lambda: os.getenv("CORS_ORIGINS", os.getenv("ALLOWED_ORIGIN", "*"))
    )
    
    # Model Configuration
    MODEL_PATH: str = Field(
        default_factory=lambda: os.getenv(
            "MODEL_PATH", 
            str(Path(__file__).resolve().parent.parent / "models" / "tusz_3class_eeg_model.h5")
        )
    )
    
    # Server Binding
    HOST: str = Field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    PORT: int = Field(default_factory=lambda: int(os.getenv("PORT", "10000")))
    
    # Data & Environment
    DATA_DIR: str = Field(default_factory=lambda: os.getenv("DATA_DIR", "data"))
    ENVIRONMENT: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))

    @property
    def cors_origins(self) -> List[str]:
        raw = os.getenv("CORS_ORIGINS") or os.getenv("ALLOWED_ORIGIN") or self.ALLOWED_ORIGIN
        if not raw or raw.strip() == "*":
            return ["*"]
        origins = [o.strip() for o in raw.split(",") if o.strip()]
        return origins if origins else ["*"]

settings = Settings()
