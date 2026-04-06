"""Event repository."""

from __future__ import annotations

import uuid
from typing import Sequence
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Event


class EventRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_batch(self, events_data: list[dict]) -> None:
        """Bulk insert events."""
        events = [Event(**data) for data in events_data]
        self.session.add_all(events)
        # Commit should be handled by caller/dependency

    async def get_by_user(
        self, user_id: uuid.UUID, limit: int = 1000, desc: bool = True
    ) -> Sequence[Event]:
        """Get events for a specific user."""
        stmt = select(Event).filter(Event.muid == user_id)
        if desc:
            stmt = stmt.order_by(Event.event_time.desc())
        else:
            stmt = stmt.order_by(Event.event_time.asc())
        
        stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def count_by_user(self, user_id: uuid.UUID) -> int:
        from sqlalchemy import func
        stmt = select(func.count(Event.event_id)).filter(Event.muid == user_id)
        result = await self.session.execute(stmt)
        return result.scalar() or 0
