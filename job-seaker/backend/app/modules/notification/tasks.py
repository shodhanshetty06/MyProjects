"""Notification module — Celery tasks.

One task, routed to the "notifications" queue (see `app/workers/celery_app.py`
`task_routes` and the `worker-email-tracking` service in `docker-compose.yml`,
which currently also serves the `email_tracking` queue):

    send_notification_email_task -> "notifications" queue

Sent as its own task (rather than inline from `NotificationService.notify`)
so a slow/throttled SES call never blocks the caller — same reasoning
`auto_apply`/`job_ingestion` use for keeping network calls in dedicated
tasks. Retry: only `TransientNotificationError` (SES throttling/5xx) is
handed to Celery's own `autoretry_for` + `retry_backoff`, matching
`auto_apply/tasks.py`'s classify-then-decide-to-retry template.
"""

from __future__ import annotations

import asyncio
import uuid
from asyncio import run as run_async

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal
from app.modules.notification.models import NotificationStatus
from app.modules.notification.repository import NotificationRepository
from app.services.notification.email_sender import SESEmailSender
from app.services.notification.errors import (
    TransientNotificationError,
    classify_ses_exception,
    is_retryable,
)
from app.services.notification.templates import render
from app.workers.base_task import BaseTask
from app.workers.celery_app import celery_app

logger = get_logger(__name__)

_settings = get_settings()


@celery_app.task(
    base=BaseTask,
    bind=True,
    name="notification.send_email",
    autoretry_for=(TransientNotificationError,),
    max_retries=_settings.NOTIFICATION_MAX_RETRIES,
)
def send_notification_email_task(
    self, notification_id: str
) -> None:  # noqa: ANN001 -- Celery `self`
    run_async(
        _send_notification_email(uuid.UUID(notification_id), retry_count=self.request.retries)
    )


async def _send_notification_email(notification_id: uuid.UUID, *, retry_count: int) -> None:
    async with AsyncSessionLocal() as session:
        repository = NotificationRepository(session)
        notification = await repository.get(notification_id)
        if notification is None:
            logger.warning(
                "Notification not found for email send",
                extra={"extra_fields": {"notification_id": str(notification_id)}},
            )
            return
        if not notification.recipient_email:
            await repository.update(
                notification,
                status=NotificationStatus.FAILED,
                error_message="No recipient_email was recorded for this notification.",
            )
            await session.commit()
            return

        subject, body_text = render(
            event_type=notification.event_type, title=notification.title, body=notification.body
        )

        try:
            sender = SESEmailSender()
            await asyncio.to_thread(
                sender.send, to=notification.recipient_email, subject=subject, body_text=body_text
            )
            await repository.update(
                notification, status=NotificationStatus.SENT, error_message=None
            )
            await session.commit()
        except Exception as exc:
            await session.rollback()
            category = classify_ses_exception(exc)
            is_last_attempt = retry_count >= _settings.NOTIFICATION_MAX_RETRIES - 1
            will_retry = is_retryable(category) and not is_last_attempt

            await repository.update(
                notification,
                status=NotificationStatus.FAILED,
                error_message=str(exc)[:2000],
            )
            await session.commit()
            logger.error(
                "Notification email send failed",
                extra={
                    "extra_fields": {
                        "notification_id": str(notification_id),
                        "category": category.value,
                        "will_retry": will_retry,
                        "error": str(exc),
                    }
                },
            )
            if will_retry:
                raise TransientNotificationError(str(exc)) from exc


__all__ = ["send_notification_email_task"]
