"""Prediction schemas."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.event import EventCreate


class PredictionRequest(BaseModel):
    """Schema for server-side prediction request."""
    user_id: uuid.UUID
    events: list[EventCreate] = Field(
        default=[], 
        description="Optional recent events not yet in the database to include in sequence."
    )


class PredictionResponse(BaseModel):
    """Schema for prediction output match task document requirements."""
    purchase_probability: float = Field(ge=0.0, le=1.0)
    churn_risk: float = Field(ge=0.0, le=1.0)
    next_event: str
    preferred_channel: str
    engagement_score: float
    inactivity_risk: float
    recommended_action: str
    active_time: str
