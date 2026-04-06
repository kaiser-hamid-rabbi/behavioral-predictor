"""Event ingestion endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, BackgroundTasks, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db_session
from app.schemas.event import EventBatchCreate
from app.services.event_service import EventService
from app.core.observability import EVENT_INGESTION_COUNT

router = APIRouter(prefix="/events", tags=["events"])


@router.post(
    "/ingest",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a batch of events"
)
async def ingest_events(
    batch: EventBatchCreate,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session)
) -> dict[str, str]:
    """
    Ingest a batch of behavioral events for processing.
    
    Returns 202 Accepted and processes the ingestion asynchronously.
    """
    event_service = EventService(session)
    
    # Normally we would put this on a real background queue (e.g. Celery / Kafka)
    # Using FastAPI background tasks for simplicity in this synchronous handler path
    background_tasks.add_task(event_service.ingest_batch, batch)
    
    EVENT_INGESTION_COUNT.inc(len(batch.events))
    
    return {"message": f"Accepted {len(batch.events)} events for processing."}
