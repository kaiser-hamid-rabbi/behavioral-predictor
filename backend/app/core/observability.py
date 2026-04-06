"""
Observability module — OpenTelemetry tracing and Prometheus metrics.

Provides:
- OpenTelemetry trace instrumentation (when enabled)
- Prometheus counter/histogram metrics for API and ML pipeline
- Health status collectors
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST

from app.core.config import get_settings

# ── Prometheus Metrics ───────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total predictions made",
    ["source"],  # "api" or "browser"
)

PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds",
    "Prediction latency in seconds",
    ["source"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
)

MODEL_VERSION_INFO = Info(
    "model_version",
    "Currently active model version",
)

TRAINING_DURATION = Histogram(
    "training_duration_seconds",
    "Model training duration in seconds",
    buckets=(60, 300, 600, 1800, 3600, 7200),
)

EVENT_INGESTION_COUNT = Counter(
    "events_ingested_total",
    "Total events ingested",
)

FEATURE_COMPUTATION_LATENCY = Histogram(
    "feature_computation_duration_seconds",
    "Feature computation latency",
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
)

ACTIVE_MODEL_SIZE_BYTES = Gauge(
    "model_size_bytes",
    "Size of the active ONNX model in bytes",
)


@contextmanager
def track_latency(histogram: Histogram, **labels: str) -> Generator[None, None, None]:
    """Context manager to track operation latency."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        histogram.labels(**labels).observe(duration)


def setup_opentelemetry() -> None:
    """Initialize OpenTelemetry tracing if enabled."""
    settings = get_settings()
    if not settings.otel_enabled:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        resource = Resource.create(
            {
                "service.name": settings.app_name,
                "service.version": "1.0.0",
                "deployment.environment": settings.app_env,
            }
        )

        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
    except ImportError:
        pass  # OpenTelemetry packages not installed — graceful degradation


def get_metrics_response() -> tuple[bytes, str]:
    """Generate Prometheus metrics response."""
    return generate_latest(), CONTENT_TYPE_LATEST
