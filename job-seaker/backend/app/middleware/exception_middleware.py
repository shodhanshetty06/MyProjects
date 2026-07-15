"""Global exception handlers.

Registered directly on the FastAPI app (not as BaseHTTPMiddleware) so they
can also catch exceptions raised inside other middleware and produce a
consistent `ErrorResponse` JSON body for every unhandled error.
"""
from __future__ import annotations

import uuid

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import AppException
from app.core.logging import get_logger
from app.schemas.common import ErrorDetail, ErrorResponse

logger = get_logger("app.exceptions")


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or str(uuid.uuid4())


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
        logger.warning(
            "handled application exception",
            extra={
                "extra_fields": {
                    "error_code": exc.error_code,
                    "status_code": exc.status_code,
                    "path": request.url.path,
                }
            },
        )
        body = ErrorResponse(
            error=ErrorDetail(
                error_code=exc.error_code, message=exc.message, details=exc.details
            ),
            request_id=_request_id(request),
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump(mode="json"))

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        body = ErrorResponse(
            error=ErrorDetail(
                error_code="validation_error",
                message="Request validation failed.",
                details={"errors": exc.errors()},
            ),
            request_id=_request_id(request),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=body.model_dump(mode="json"),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        body = ErrorResponse(
            error=ErrorDetail(
                error_code="http_error",
                message=str(exc.detail),
            ),
            request_id=_request_id(request),
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump(mode="json"))

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "unhandled exception",
            exc_info=exc,
            extra={"extra_fields": {"path": request.url.path}},
        )
        body = ErrorResponse(
            error=ErrorDetail(
                error_code="internal_error",
                message="An unexpected error occurred.",
            ),
            request_id=_request_id(request),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=body.model_dump(mode="json"),
        )
