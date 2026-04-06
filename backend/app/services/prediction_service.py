"""Prediction service."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ModelNotReadyError
from app.db.repositories.model_repository import ModelRepository
from app.schemas.prediction import PredictionRequest, PredictionResponse


class PredictionService:
    def __init__(self, session: AsyncSession, redis_client=None) -> None:
        self.session = session
        self.redis_client = redis_client
        self.model_repo = ModelRepository(session)
        # Note: In a real system, the model predictor instance itself is kept in memory
        # to avoid reloading large files per request. See app.ml.inference.predictor 

    async def predict_for_user(self, request: PredictionRequest) -> PredictionResponse:
        """
        Produce predictions for a user given their sequence of events.
        """
        # 1. Fetch active model version details to know Which ONNX/PT file to use
        active_model = await self.model_repo.get_active()
        if not active_model:
            raise ModelNotReadyError("No active model available.")

        # Realistically, this delegates to the loaded ML predictor.
        # For layout purposes, creating a mock returning the correct signature schema
        from app.ml.inference.predictor import get_predictor
        predictor = get_predictor()
        if not predictor:
             raise ModelNotReadyError("Predictor engine not initialized.")

        # We would merge request.events with past events from event_repo if needed
        # Demonstrate hitting the Redis Online Feature Store for real-time user context:
        from app.services.feature_service import FeatureService
        feature_service = FeatureService(self.session, self.redis_client)
        online_features = await feature_service.get_user_features(request.user_id)
        
        # Real logic injects online_features alongside events into the predictor
        prediction_result = await predictor.predict(request.user_id, request.events)
        return PredictionResponse(**prediction_result)
