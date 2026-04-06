"""User repository."""

from __future__ import annotations

import uuid
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, user_id: uuid.UUID) -> User | None:
        """Get a user by ID."""
        stmt = select(User).filter(User.id == user_id)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def create(self, user_data: dict) -> User:
        """Create a new user."""
        user = User(**user_data)
        self.session.add(user)
        return user

    async def create_batch(self, users_data: list[dict]) -> None:
        """Bulk insert users."""
        users = [User(**data) for data in users_data]
        self.session.add_all(users)

    async def get_all(self, limit: int = 1000, offset: int = 0) -> Sequence[User]:
        """Get paginated users."""
        stmt = select(User).order_by(User.id).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return result.scalars().all()
