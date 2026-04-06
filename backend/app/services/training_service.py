"""Training service."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.model import TrainingRequest, TrainingResponse


class TrainingService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def trigger_training(self, request: TrainingRequest) -> TrainingResponse:
        """
        Trigger an asynchronous model training job via Celery worker.
        """
        # We import the celery task delay here to avoid circular imports.
        # Assuming app.workers.tasks.training_task has a 'run_training_pipeline' task
        try:
            from app.workers.tasks.training_task import run_training_pipeline
            
            task = run_training_pipeline.delay(
                epochs=request.epochs,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                force_retrain=request.force_retrain,
                sample_users=request.sample_users
            )
            return TrainingResponse(
                task_id=task.id,
                message="Training job successfully queued.",
                status="queued"
            )
        except Exception as e:
            # Optionally fall back or re-raise TrainingError
            from app.core.exceptions import TrainingError
            raise TrainingError(f"Could not trigger training: {str(e)}")
