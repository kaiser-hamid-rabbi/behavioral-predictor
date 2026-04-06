"""Prediction endpoints."""

from __future__ import annotations

import uuid
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db_session, get_redis
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_service import PredictionService
from app.core.observability import PREDICTION_COUNT, PREDICTION_LATENCY, track_latency

router = APIRouter(prefix="/predict", tags=["predictions"])


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Generate predictions for a user"
)
async def predict(
    request: PredictionRequest,
    session: AsyncSession = Depends(get_db_session),
    redis = Depends(get_redis)
) -> PredictionResponse:
    """
    Given a user and recent events, returns real-time behavioral predictions.
    """
    service = PredictionService(session, redis_client=redis)
    
    with track_latency(PREDICTION_LATENCY, source="api"):
        prediction = await service.predict_for_user(request)
        
    PREDICTION_COUNT.labels(source="api").inc()
    
    return prediction
