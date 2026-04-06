"""Model and Training schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ModelVersionResponse(BaseModel):
    """Model version representation."""
    id: int
    version: str
    model_path: str
    onnx_path: str | None = None
    metrics: dict | None = None
    config: dict | None = None
    model_size_bytes: int | None = None
    is_active: bool
    created_at: datetime


class TrainingRequest(BaseModel):
    """Request schema for triggering a training job."""
    epochs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    force_retrain: bool = False
    sample_users: int | None = None


class TrainingResponse(BaseModel):
    """Response when a training job is queued."""
    task_id: str
    message: str
    status: str = "queued"
