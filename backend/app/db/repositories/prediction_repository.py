"""Prediction Log repository."""

from __future__ import annotations

import uuid
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import PredictionLog


class PredictionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def log_prediction(self, data: dict) -> PredictionLog:
        """Create a new prediction log entry."""
        log = PredictionLog(**data)
        self.session.add(log)
        return log

    async def get_by_user(self, user_id: uuid.UUID, limit: int = 100) -> Sequence[PredictionLog]:
        """Get prediction logs for a user."""
        stmt = select(PredictionLog).filter(PredictionLog.user_id == user_id).order_by(PredictionLog.created_at.desc()).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()
