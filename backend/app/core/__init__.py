"""
Configuration module — single source of truth for all application settings.

Uses pydantic-settings to load from environment variables with .env fallback.
All settings are typed, validated, and documented.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────
    app_name: str = "behavioral-predictor"
    app_env: str = "development"
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: str = "INFO"

    # ── Database (PostgreSQL) ────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/behavioral_predictor"
    database_pool_size: int = 20
    database_max_overflow: int = 10

    # ── Redis ────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Celery ───────────────────────────────────────────────────────────
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── ML Training Hyperparameters ──────────────────────────────────────
    ml_model_dir: str = "./models"
    ml_batch_size: int = 256
    ml_epochs: int = 50
    ml_learning_rate: float = 0.001
    ml_sequence_length: int = 20
    ml_embedding_dim: int = 32
    ml_transformer_heads: int = 4
    ml_transformer_layers: int = 2
    ml_transformer_ff_dim: int = 128
    ml_dropout: float = 0.1

    # ── Model Compression ────────────────────────────────────────────────
    compression_quantize: bool = True
    compression_prune: bool = True
    compression_prune_amount: float = 0.3
    compression_target_size_mb: float = 1.0

    # ── Data ─────────────────────────────────────────────────────────────
    data_dir: str = "../dataset/EventsData"
    data_sample_users: int = 5000
    data_chunk_size: int = 50000

    # ── Observability ────────────────────────────────────────────────────
    otel_enabled: bool = False
    otel_exporter_endpoint: str = "http://localhost:4317"
    prometheus_enabled: bool = True

    # ── CORS ─────────────────────────────────────────────────────────────
    cors_origins: str = '["http://localhost:3000","http://localhost:8000"]'

    @field_validator("app_log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v_upper

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from JSON string."""
        if isinstance(self.cors_origins, list):
            return self.cors_origins
        try:
            return json.loads(self.cors_origins)
        except (json.JSONDecodeError, TypeError):
            return ["http://localhost:3000"]

    @property
    def model_dir_path(self) -> Path:
        """Resolved model directory path."""
        return Path(self.ml_model_dir).resolve()

    @property
    def data_dir_path(self) -> Path:
        """Resolved data directory path."""
        return Path(self.data_dir).resolve()

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
