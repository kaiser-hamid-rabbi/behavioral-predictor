"""Training trigger endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db_session
from app.schemas.model import TrainingRequest, TrainingResponse
from app.services.training_service import TrainingService

router = APIRouter(prefix="/train", tags=["training"])


@router.post(
    "",
    response_model=TrainingResponse,
    summary="Trigger model training job"
)
async def trigger_training(
    request: TrainingRequest,
    session: AsyncSession = Depends(get_db_session)
) -> TrainingResponse:
    """
    Dispatches a Celery background task to run the complete training,
    compression, and registration pipeline.
    """
    service = TrainingService(session)
    return await service.trigger_training(request)
