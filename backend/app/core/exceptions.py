"""
Custom exception hierarchy with structured error responses.

All application exceptions inherit from AppError for consistent
error handling in the FastAPI exception handler middleware.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base application error."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class NotFoundError(AppError):
    """Resource not found."""

    def __init__(self, resource: str, identifier: str) -> None:
        super().__init__(
            message=f"{resource} not found: {identifier}",
            status_code=404,
            error_code="NOT_FOUND",
            details={"resource": resource, "identifier": identifier},
        )


class ValidationError(AppError):
    """Input validation failed."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            status_code=422,
            error_code="VALIDATION_ERROR",
            details=details or {},
        )


class ModelNotReadyError(AppError):
    """ML model is not loaded or available."""

    def __init__(self, message: str = "Model is not ready for inference") -> None:
        super().__init__(
            message=message,
            status_code=503,
            error_code="MODEL_NOT_READY",
        )


class TrainingError(AppError):
    """Error during model training."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            status_code=500,
            error_code="TRAINING_ERROR",
            details=details or {},
        )


class DatabaseError(AppError):
    """Database operation failed."""

    def __init__(self, message: str = "Database operation failed") -> None:
        super().__init__(
            message=message,
            status_code=500,
            error_code="DATABASE_ERROR",
        )


class FeatureStoreError(AppError):
    """Feature computation or retrieval failed."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            status_code=500,
            error_code="FEATURE_STORE_ERROR",
            details=details or {},
        )
