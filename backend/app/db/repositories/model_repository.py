"""Model Version repository."""

from __future__ import annotations

from typing import Sequence

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ModelVersion


class ModelRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_active(self) -> ModelVersion | None:
        """Get the currently active model version."""
        stmt = select(ModelVersion).filter(ModelVersion.is_active == True)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_by_version(self, version: str) -> ModelVersion | None:
        """Get model by version string."""
        stmt = select(ModelVersion).filter(ModelVersion.version == version)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def create(self, data: dict) -> ModelVersion:
        """Register a new model version."""
        model = ModelVersion(**data)
        self.session.add(model)
        return model

    async def set_active(self, version: str) -> ModelVersion | None:
        """Set a specific version as active, deactivating others."""
        # Deactivate all
        await self.session.execute(
            update(ModelVersion).values(is_active=False)
        )
        # Activate target
        stmt = update(ModelVersion).where(ModelVersion.version == version).values(is_active=True).returning(ModelVersion)
        result = await self.session.execute(stmt)
        return result.scalars().first()
    
    async def get_all(self, limit: int = 100) -> Sequence[ModelVersion]:
        stmt = select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()
