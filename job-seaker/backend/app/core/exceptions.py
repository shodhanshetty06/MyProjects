"""Application exception hierarchy.

Every domain/module-level error should subclass `AppException` so the global
exception handler (see `app/middleware/exception_middleware.py`) can map it
to a consistent JSON error response without each router needing its own
try/except boilerplate.
"""
from __future__ import annotations

from typing import Any


class AppException(Exception):
    """Base class for all application-raised, handled exceptions."""

    status_code: int = 500
    error_code: str = "internal_error"
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        error_code: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.message
        self.error_code = error_code or self.error_code
        self.status_code = status_code or self.status_code
        self.details = details or {}
        super().__init__(self.message)


class NotFoundError(AppException):
    status_code = 404
    error_code = "not_found"
    message = "The requested resource was not found."


class ValidationError(AppException):
    status_code = 422
    error_code = "validation_error"
    message = "Validation failed."


class UnauthorizedError(AppException):
    status_code = 401
    error_code = "unauthorized"
    message = "Authentication is required."


class ForbiddenError(AppException):
    status_code = 403
    error_code = "forbidden"
    message = "You do not have permission to perform this action."


class ConflictError(AppException):
    status_code = 409
    error_code = "conflict"
    message = "The request conflicts with the current state of the resource."


class RateLimitedError(AppException):
    status_code = 429
    error_code = "rate_limited"
    message = "Too many requests."


class ServiceUnavailableError(AppException):
    status_code = 503
    error_code = "service_unavailable"
    message = "A dependent service is currently unavailable."


class TransientProviderError(ServiceUnavailableError):
    """Retryable AI/embedding provider failure: connection refused, timeout,
    or a 5xx response. Raised by `providers/ai/*` and `providers/embedding/*`
    so `AIService`/`EmbeddingService` know it's worth retrying."""

    error_code = "provider_transient_error"
    message = "The AI provider is temporarily unavailable."


class ProviderUnavailableError(ServiceUnavailableError):
    """Non-retryable AI/embedding provider failure: bad configuration, model
    refusal, invalid/unparseable response, or all fallback models exhausted.
    Raised by `providers/ai/*` and `providers/embedding/*`."""

    error_code = "provider_unavailable"
    message = "The AI provider is unavailable."
