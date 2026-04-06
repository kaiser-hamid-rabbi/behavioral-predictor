"""Feature engineering celery tasks."""

from __future__ import annotations

from app.workers.celery_app import celery_app
import time

@celery_app.task(bind=True, name="tasks.features.compute_batch")
def compute_features_batch(self, user_ids: list[str]) -> dict[str, int]:
    """
    Background task to compute aggregate features for a batch of users 
    and store them into Postgres feature store.
    """
    self.update_state(state="PROGRESS", meta={"processed": 0, "total": len(user_ids)})
    
    # Mock computation
    time.sleep(1)
    
    return {"processed": len(user_ids)}
