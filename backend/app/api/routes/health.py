"""System health check endpoint."""

from __future__ import annotations

import time
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.dependencies import get_db_session, get_redis
from app.schemas.health import HealthResponse, ComponentHealth
from app.db.repositories.model_repository import ModelRepository

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check(
    session: AsyncSession = Depends(get_db_session),
    redis = Depends(get_redis) # Redis client
) -> HealthResponse:
    """
    Comprehensive system health check.
    Verifies DB connection, Redis reachability, and model loaded status.
    """
    components = {}
    system_status = "ok"

    # 1. Check Database
    db_start = time.perf_counter()
    try:
        await session.execute(text("SELECT 1"))
        components["database"] = ComponentHealth(
            status="ok", 
            latency_ms=(time.perf_counter() - db_start) * 1000
        )
    except Exception as e:
        components["database"] = ComponentHealth(status="error", details=str(e))
        system_status = "degraded"

    # 2. Check Redis
    redis_start = time.perf_counter()
    try:
        if redis:
            await redis.ping()
            components["redis"] = ComponentHealth(
                status="ok",
                latency_ms=(time.perf_counter() - redis_start) * 1000
            )
        else:
            components["redis"] = ComponentHealth(status="unavailable", details="Redis client not initialized")
    except Exception as e:
        components["redis"] = ComponentHealth(status="error", details=str(e))
        system_status = "degraded"

    # 3. Check ML Model
    try:
        repo = ModelRepository(session)
        active = await repo.get_active()
        if active:
            components["model_registry"] = ComponentHealth(
                status="ok", 
                details=f"Active version: {active.version}"
            )
        else:
            components["model_registry"] = ComponentHealth(
                status="warning", 
                details="No active model deployed"
            )
    except Exception as e:
        components["model_registry"] = ComponentHealth(status="error", details=str(e))

    return HealthResponse(
        status=system_status,
        version="1.0.0",
        components=components
    )
