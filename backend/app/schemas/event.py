"""Event schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, HttpUrl


class EventCreate(BaseModel):
    """Schema for incoming single event."""
    
    event_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    muid: uuid.UUID
    session_id: uuid.UUID | None = None
    event_name: str = Field(..., max_length=50)
    event_time: datetime = Field(default_factory=lambda: datetime.now().astimezone())
    device_os: str | None = Field(default=None, max_length=20)
    channel: str | None = Field(default=None, max_length=20)
    traffic_source: str | None = Field(default=None, max_length=20)
    category: str | None = Field(default=None, max_length=50)
    product_id: str | None = Field(default=None, max_length=100)
    page_url: str | None = None


class EventBatchCreate(BaseModel):
    """Schema for incoming batch of events."""
    
    events: list[EventCreate] = Field(..., min_length=1, max_length=1000)


class EventResponse(EventCreate):
    """Schema for event response."""
    pass
