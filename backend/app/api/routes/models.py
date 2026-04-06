"""Model Registry and Browser download endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db_session
from app.schemas.model import ModelVersionResponse
from app.db.repositories.model_repository import ModelRepository
import os

router = APIRouter(prefix="/models", tags=["models"])


@router.get(
    "/active",
    response_model=ModelVersionResponse,
    summary="Get active model details"
)
async def get_active_model(
    session: AsyncSession = Depends(get_db_session)
) -> ModelVersionResponse:
    """Retrieve metadata about the currently deployed model version."""
    repo = ModelRepository(session)
    active = await repo.get_active()
    if not active:
        raise HTTPException(status_code=404, detail="No active model found.")
    return active  # type: ignore[return-value]


@router.get(
    "/download",
    response_class=FileResponse,
    summary="Download ONNX model for browser inference"
)
async def download_model(
    session: AsyncSession = Depends(get_db_session)
) -> FileResponse:
    """Download the actual active ONNX model file."""
    repo = ModelRepository(session)
    active = await repo.get_active()
    
    if not active or not active.onnx_path or not os.path.exists(active.onnx_path):
        raise HTTPException(status_code=404, detail="Model file not available.")
        
    return FileResponse(
        active.onnx_path,
        media_type="application/octet-stream",
        filename="behavioral_predictor.onnx"
    )
