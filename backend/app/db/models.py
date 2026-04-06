"""
SQLAlchemy ORM models for the behavioral prediction system.

Tables:
- users: User profiles
- events: User behavioral events (partition-ready by event_time)
- user_features: Computed feature vectors per user (feature store)
- model_versions: ML model registry
- prediction_logs: Prediction audit trail for feedback loop
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class User(Base):
    """User profile table."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    location: Mapped[str | None] = mapped_column(String(100), nullable=True)
    last_active_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_users_location", "location"),
        Index("ix_users_last_active", "last_active_at"),
    )


class Event(Base):
    """
    User behavioral event table.

    Designed for PostgreSQL range partitioning on event_time.
    The composite primary key (event_id, event_time) enables partition pruning.
    """

    __tablename__ = "events"

    event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    muid: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    event_name: Mapped[str] = mapped_column(String(50), nullable=False)
    event_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )
    device_os: Mapped[str | None] = mapped_column(String(20), nullable=True)
    channel: Mapped[str | None] = mapped_column(String(20), nullable=True)
    traffic_source: Mapped[str | None] = mapped_column(String(20), nullable=True)
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    product_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    page_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_events_muid_time", "muid", "event_time"),
        Index("ix_events_session", "session_id"),
        Index("ix_events_name", "event_name"),
        Index("ix_events_category", "category"),
        {
            "postgresql_partition_by": "RANGE (event_time)",
            "comment": "Partitioned by month on event_time for scalable querying",
        },
    )


class UserFeature(Base):
    """
    Computed feature vectors per user — acts as the feature store.

    Features are stored as JSONB for schema flexibility as feature
    engineering evolves. feature_version tracks compatibility.
    """

    __tablename__ = "user_features"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True
    )
    features: Mapped[dict] = mapped_column(JSONB, nullable=False)
    feature_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_user_features_version", "feature_version"),
    )


class ModelVersion(Base):
    """
    ML model registry — tracks all trained model versions.

    Stores model metadata, paths, evaluation metrics, and
    the active flag for serving.
    """

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    model_path: Mapped[str] = mapped_column(Text, nullable=False)
    onnx_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    model_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_model_versions_active", "is_active"),
    )


class PredictionLog(Base):
    """
    Prediction audit log for feedback loop and model drift detection.

    Stores the input sequence, predictions, and (optionally) actual outcomes
    for retraining and evaluation.
    """

    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    input_sequence: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    predictions: Mapped[dict] = mapped_column(JSONB, nullable=False)
    actual_outcome: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_prediction_logs_user", "user_id"),
        Index("ix_prediction_logs_model", "model_version"),
        Index("ix_prediction_logs_created", "created_at"),
    )
