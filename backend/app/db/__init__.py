"""
Async SQLAlchemy 2.0 engine and session factory.

Configures connection pooling and provides the base declarative class
for all ORM models.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


def _create_engine():  # type: ignore[no-untyped-def]
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=settings.app_debug,
    )


engine = _create_engine()

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_engine():  # type: ignore[no-untyped-def]
    """Return the async engine (useful for raw connections)."""
    return engine


async def dispose_engine() -> None:
    """Dispose the engine on shutdown."""
    await engine.dispose()
