"""
FastAPI dependency injection providers.

Provides request-scoped database sessions, Redis connections,
and shared service instances via FastAPI's Depends() system.
"""

from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.base import async_session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield a request-scoped async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_redis():  # type: ignore[no-untyped-def]
    """Get Redis connection (lazy import to handle missing redis gracefully)."""
    try:
        import redis.asyncio as aioredis

        settings = get_settings()
        client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        try:
            yield client
        finally:
            await client.aclose()
    except ImportError:
        yield None
    except Exception:
        yield None


def get_config() -> Settings:
    """Inject application configuration."""
    return get_settings()
