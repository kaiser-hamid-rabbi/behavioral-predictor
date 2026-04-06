"""Event service."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.event_repository import EventRepository
from app.streaming.kafka_producer import kafka_producer
from app.schemas.event import EventBatchCreate


class EventService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def ingest_batch(self, batch: EventBatchCreate) -> None:
        """
        Stream events directly to Kafka broker for ultimate high-throughput scalability.
        Bypassing standard PostgreSQL inserts solves DB connection saturations.
        """
        for event in batch.events:
            # Prepare payload for streaming network
            payload = event.model_dump()
            payload["muid"] = str(batch.muid)
            
            # Fire and forget onto the raw-events topic
            await kafka_producer.publish_event(
                topic="raw_behavioral_events",
                payload=payload,
                key=str(batch.muid)
            )
        return len(batch.events)
