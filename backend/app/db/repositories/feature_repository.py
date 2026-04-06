"""User Feature repository."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from app.db.models import UserFeature


class FeatureRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, user_id: uuid.UUID) -> UserFeature | None:
        """Get features for a user."""
        stmt = select(UserFeature).filter(UserFeature.user_id == user_id)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def upsert(self, user_id: uuid.UUID, features: dict, version: int, computed_at) -> UserFeature:
        """Insert or update user features."""
        stmt = insert(UserFeature).values(
            user_id=user_id,
            features=features,
            feature_version=version,
            computed_at=computed_at,
        ).on_conflict_do_update(
            index_elements=['user_id'],
            set_={
                'features': features,
                'feature_version': version,
                'computed_at': computed_at,
            }
        ).returning(UserFeature)
        
        result = await self.session.execute(stmt)
        return result.scalars().first()
