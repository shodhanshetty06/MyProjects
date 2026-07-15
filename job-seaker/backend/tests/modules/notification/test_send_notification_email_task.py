"""Unit tests for `_send_notification_email`'s retry-classification branch —
same shape as `auto_apply`'s exception-handling tests: a transient SES
failure should raise `TransientNotificationError` (picked up by Celery's
`autoretry_for`) while a permanent one should not. `SESEmailSender` is
monkeypatched at the module level, so no real AWS credentials are required.

Needs a reachable Postgres with migrations applied:

    docker compose up -d postgres redis
    docker compose exec api alembic upgrade head
    docker compose exec api pytest tests/modules/notification/test_send_notification_email_task.py
"""

from __future__ import annotations

import uuid

import pytest

from app.db.session import AsyncSessionLocal
from app.modules.notification import tasks as notification_tasks
from app.modules.notification.models import (
    Notification,
    NotificationChannel,
    NotificationEventType,
    NotificationStatus,
)
from app.services.notification.errors import TransientNotificationError


async def _seed_pending_notification(*, recipient_email: str | None) -> uuid.UUID:
    async with AsyncSessionLocal() as session:
        notification = Notification(
            user_id=uuid.uuid4(),
            event_type=NotificationEventType.APPLICATION_STATUS_CHANGED,
            channel=NotificationChannel.EMAIL,
            title="Application status updated",
            body="Your application moved to interview.",
            recipient_email=recipient_email,
            data={},
            status=NotificationStatus.PENDING,
        )
        session.add(notification)
        await session.commit()
        await session.refresh(notification)
        return notification.id


async def _fetch_notification(notification_id: uuid.UUID) -> Notification:
    async with AsyncSessionLocal() as session:
        return await session.get(Notification, notification_id)


class _RaisingSender:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def send(self, **kwargs):
        raise self._error


class _SucceedingSender:
    def send(self, **kwargs):
        return "ses-message-id"


async def test_transient_failure_raises_and_marks_failed_before_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notification_id = await _seed_pending_notification(recipient_email="user@example.com")
    monkeypatch.setattr(
        notification_tasks,
        "SESEmailSender",
        lambda: _RaisingSender(TransientNotificationError("throttled")),
    )

    with pytest.raises(TransientNotificationError):
        await notification_tasks._send_notification_email(notification_id, retry_count=0)

    notification = await _fetch_notification(notification_id)
    assert notification.status == NotificationStatus.FAILED
    assert notification.error_message


async def test_transient_failure_on_last_attempt_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notification_id = await _seed_pending_notification(recipient_email="user@example.com")
    monkeypatch.setattr(
        notification_tasks,
        "SESEmailSender",
        lambda: _RaisingSender(TransientNotificationError("throttled")),
    )

    # retry_count at NOTIFICATION_MAX_RETRIES - 1 is the last attempt — should
    # not re-raise (Celery would otherwise retry past its own max_retries).
    max_retries = notification_tasks._settings.NOTIFICATION_MAX_RETRIES
    await notification_tasks._send_notification_email(notification_id, retry_count=max_retries - 1)

    notification = await _fetch_notification(notification_id)
    assert notification.status == NotificationStatus.FAILED


async def test_missing_recipient_email_fails_without_calling_sender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notification_id = await _seed_pending_notification(recipient_email=None)
    calls: list[str] = []
    monkeypatch.setattr(
        notification_tasks, "SESEmailSender", lambda: calls.append("called") or _SucceedingSender()
    )

    await notification_tasks._send_notification_email(notification_id, retry_count=0)

    assert calls == []
    notification = await _fetch_notification(notification_id)
    assert notification.status == NotificationStatus.FAILED


async def test_successful_send_marks_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    notification_id = await _seed_pending_notification(recipient_email="user@example.com")
    monkeypatch.setattr(notification_tasks, "SESEmailSender", lambda: _SucceedingSender())

    await notification_tasks._send_notification_email(notification_id, retry_count=0)

    notification = await _fetch_notification(notification_id)
    assert notification.status == NotificationStatus.SENT
    assert notification.error_message is None
