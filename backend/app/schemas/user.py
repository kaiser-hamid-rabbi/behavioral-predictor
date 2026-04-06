"""User schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    """Schema for creating a user."""
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    email: EmailStr | None = None
    phone: str | None = Field(None, max_length=50)
    name: str | None = Field(None, max_length=255)
    location: str | None = Field(None, max_length=100)
    last_active_at: datetime | None = None


class UserResponse(UserCreate):
    """User response schema."""
    created_at: datetime
    updated_at: datetime


class UserFeatureResponse(BaseModel):
    """Schema for retrieving a user's features."""
    user_id: uuid.UUID
    features: dict
    feature_version: int
    computed_at: datetime
