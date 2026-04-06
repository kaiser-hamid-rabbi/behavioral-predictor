"""Feature service."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.feature_repository import FeatureRepository
from app.schemas.user import UserFeatureResponse
from app.core.config import get_settings


class FeatureService:
    def __init__(self, session: AsyncSession, redis_client = None):
        self.session = session
        self.redis = redis_client

    async def get_user_features(self, user_id: uuid.UUID) -> dict[str, Any] | None:
        """
        Retrieve features. Checks Redis (Online Feature Store) first for 
        sub-millisecond latency. Falls back to PostgreSQL (Offline Store).
        """
        cache_key = f"features:user:{user_id}"
        
        # 1. Attempt sub-millisecond Redis read
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
        
        # 2. Fallback to offline Postgres
        repo = FeatureRepository(self.session)
        features_record = await repo.get_latest_by_user(user_id)
        
        if features_record and features_record.features:
            payload = features_record.features
            
            # 3. Populate cache behind the scenes (Write-through)
            if self.redis:
                await self.redis.set(cache_key, json.dumps(payload), ex=3600)  # 1 hr TTL
                
            return payload
            
        return None
