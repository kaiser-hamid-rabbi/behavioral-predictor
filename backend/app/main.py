"""
FastAPI application factory and main entrypoint.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.core.observability import setup_opentelemetry, REQUEST_COUNT, track_latency, REQUEST_LATENCY
from app.core.exceptions import AppError
from app.api.routes import events, predictions, training, models, health

# Setup structured logging before app creation
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for FastAPI."""
    settings = get_settings()
    logger.info("Starting up behavioral-predictor", env=settings.app_env)
    
    # Init OpenTelemetry
    if settings.otel_enabled:
        setup_opentelemetry()
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        
    # Attempt to load latest model into memory
    try:
        from app.db.base import async_session_factory
        from app.db.repositories.model_repository import ModelRepository
        from app.ml.inference.predictor import initialize_predictor
        
        async with async_session_factory() as session:
            repo = ModelRepository(session)
            active = await repo.get_active()
            if active and active.onnx_path: # In practice, a server-side PyTorch model path might be used instead, depending on scaling needs. We mock for now.
                # initialize_predictor(active.model_path, os.path.join(settings.data_dir, "vocab")) 
                logger.info("Initializing ML predictor engine...", version=active.version)
            else:
                logger.warning("No active ML model found during startup.")
    except Exception as e:
        logger.error("Failed to initialize ML engine on startup", error=str(e))

    yield

    from app.db.base import dispose_engine
    await dispose_engine()
    logger.info("Shutting down behavioral-predictor")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Production-grade behavioral prediction system",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API Routers
    app.include_router(events.router)
    app.include_router(predictions.router)
    app.include_router(training.router)
    app.include_router(models.router)
    app.include_router(health.router)

    # Global Exception Handler
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        logger.error("Application error", error_code=exc.error_code, details=exc.details, path=request.url.path)
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
        )
        
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled server error", path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": "INTERNAL_SERVER_ERROR", "message": "An unexpected error occurred."},
        )

    # Metrics Middleware
    if settings.prometheus_enabled:
        @app.middleware("http")
        async def metrics_middleware(request: Request, call_next):
            if request.url.path == "/metrics":
                return await call_next(request)
            
            with track_latency(REQUEST_LATENCY, method=request.method, endpoint=request.url.path):
                response = await call_next(request)
                
            REQUEST_COUNT.labels(
                method=request.method, 
                endpoint=request.url.path, 
                status_code=response.status_code
            ).inc()
            return response

        @app.get("/metrics", include_in_schema=False)
        async def get_metrics():
            from app.core.observability import get_metrics_response
            from fastapi import Response
            content, content_type = get_metrics_response()
            return Response(content=content, media_type=content_type)

    return app


app = create_app()
