"""Base Celery task with structured logging and consistent error handling.

Module tasks should subclass `BaseTask` (via `@celery_app.task(base=BaseTask)`)
instead of the default `celery.Task` so failures are logged uniformly and
future cross-cutting behavior (metrics, tracing, retry policy) can be added
in one place.
"""
from __future__ import annotations

from typing import Any

from celery import Task

from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseTask(Task):
    """Common base class for all Celery tasks in the platform."""

    autoretry_for: tuple[type[Exception], ...] = ()
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True
    max_retries = 3

    def on_failure(
        self, exc: Exception, task_id: str, args: Any, kwargs: Any, einfo: Any
    ) -> None:
        logger.error(
            "Task failed",
            extra={
                "extra_fields": {
                    "task_name": self.name,
                    "task_id": task_id,
                    "error": str(exc),
                }
            },
        )
        super().on_failure(exc, task_id, args, kwargs, einfo)

    def on_success(self, retval: Any, task_id: str, args: Any, kwargs: Any) -> None:
        logger.info(
            "Task completed",
            extra={"extra_fields": {"task_name": self.name, "task_id": task_id}},
        )
        super().on_success(retval, task_id, args, kwargs)
