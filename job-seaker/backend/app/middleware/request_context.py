"""Injects a request-scoped request ID into every response and log record."""
from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.constants import REQUEST_ID_HEADER
from app.core.logging import request_id_ctx_var


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns (or propagates) a unique request ID for tracing."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        token = request_id_ctx_var.set(request_id)
        request.state.request_id = request_id

        try:
            response = await call_next(request)
        finally:
            request_id_ctx_var.reset(token)

        response.headers[REQUEST_ID_HEADER] = request_id
        return response
