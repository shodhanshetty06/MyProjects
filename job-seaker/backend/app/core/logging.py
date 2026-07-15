"""Structured logging configuration.

Uses stdlib `logging` with a JSON formatter so logs are directly consumable
by log aggregators (CloudWatch, ELK, Datadog, etc.) in every environment.
A `request_id` contextvar is injected into every record so requests can be
traced end-to-end across middleware, services, and Celery tasks.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from app.core.config import get_settings

request_id_ctx_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


class JSONFormatter(logging.Formatter):
    """Renders each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        request_id = request_id_ctx_var.get()
        if request_id:
            payload["request_id"] = request_id

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)

        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Configure the root logger once, at application startup."""
    settings = get_settings()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(settings.LOG_LEVEL.upper())

    # Quiet noisy third-party loggers by default.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
