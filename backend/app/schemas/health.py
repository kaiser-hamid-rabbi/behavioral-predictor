"""Health check schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ComponentHealth(BaseModel):
    """Health status of an individual component."""
    status: str
    details: str | None = None
    latency_ms: float | None = None


class HealthResponse(BaseModel):
    """Comprehensive system health response."""
    status: str
    version: str
    components: dict[str, ComponentHealth]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "version": "1.0.0",
                "components": {
                    "database": {"status": "ok", "latency_ms": 12.3},
                    "redis": {"status": "ok", "latency_ms": 1.1},
                    "model_registry": {"status": "ok", "details": "Active version: v1"}
                }
            }
        }
    )
