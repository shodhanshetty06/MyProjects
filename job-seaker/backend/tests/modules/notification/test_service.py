"""Integration tests for `NotificationService` against a real DB session,
following the same `AsyncSessionLocal` seeding convention as other modules'
tests. The EMAIL channel's Celery enqueue (`notification.send_email.delay`)
requires a reachable Redis broker — same requirement the rest of the test
suite already has.

Needs a reachable Postgres + Redis with migrations applied:

    docker compose up -d postgres redis
    docker compose exec api alembic upgrade head
    docker compose exec api pytest tests/modules/notification/test_service.py
"""

from __future__ import annotations

import uuid

from app.db.session import AsyncSessionLocal
from app.modules.notification.models import (
    NotificationChannel,
    NotificationEventType,
    NotificationStatus,
)
from app.modules.notification.repository import (
    NotificationPreferenceRepository,
    NotificationRepository,
)
from app.modules.notification.service import NotificationService


def _build_service(session) -> NotificationService:
    return NotificationService(
        repository=NotificationRepository(session),
        preference_repository=NotificationPreferenceRepository(session),
    )


async def test_notify_uses_default_channels_when_no_preference_rows_exist() -> None:
    user_id = uuid.uuid4()

    async with AsyncSessionLocal() as session:
        service = _build_service(session)
        notifications = await service.notify(
            user_id=user_id,
            event_type=NotificationEventType.APPLICATION_STATUS_CHANGED,
            title="Status updated",
            body="Your application moved to interview.",
            recipient_email="user@example.com",
        )
        await session.commit()

    channels = {n.channel for n in notifications}
    # NOTIFICATION_DEFAULT_CHANNELS defaults to "in_app,email"
    assert channels == {NotificationChannel.IN_APP, NotificationChannel.EMAIL}
    in_app = next(n for n in notifications if n.channel == NotificationChannel.IN_APP)
    assert in_app.status == NotificationStatus.SENT
    email = next(n for n in notifications if n.channel == NotificationChannel.EMAIL)
    assert email.status == NotificationStatus.PENDING


async def test_notify_skips_email_without_recipient_address() -> None:
    user_id = uuid.uuid4()

    async with AsyncSessionLocal() as session:
        service = _build_service(session)
        notifications = await service.notify(
            user_id=user_id,
            event_type=NotificationEventType.APPLICATION_STATUS_CHANGED,
            title="Status updated",
            body="Your application moved to interview.",
            recipient_email=None,
        )
        await session.commit()

    channels = {n.channel for n in notifications}
    assert NotificationChannel.EMAIL not in channels
    assert NotificationChannel.IN_APP in channels


async def test_notify_respects_disabled_channel_preference() -> None:
    user_id = uuid.uuid4()

    async with AsyncSessionLocal() as session:
        preference_repo = NotificationPreferenceRepository(session)
        await preference_repo.upsert(
            user_id=user_id,
            event_type=NotificationEventType.APPLICATION_STATUS_CHANGED,
            channel=NotificationChannel.IN_APP,
            enabled=False,
        )
        await session.commit()

    async with AsyncSessionLocal() as session:
        service = _build_service(session)
        notifications = await service.notify(
            user_id=user_id,
            event_type=NotificationEventType.APPLICATION_STATUS_CHANGED,
            title="Status updated",
            body="Body",
            recipient_email="user@example.com",
        )
        await session.commit()

    channels = {n.channel for n in notifications}
    assert NotificationChannel.IN_APP not in channels
    assert NotificationChannel.EMAIL in channels


async def test_get_preferences_returns_full_matrix_with_defaults() -> None:
    user_id = uuid.uuid4()

    async with AsyncSessionLocal() as session:
        service = _build_service(session)
        preferences = await service.get_preferences(user_id=user_id)

    expected_count = len(NotificationEventType) * len(NotificationChannel)
    assert len(preferences) == expected_count
    in_app_status_changed = next(
        p
        for p in preferences
        if p.event_type == NotificationEventType.APPLICATION_STATUS_CHANGED
        and p.channel == NotificationChannel.IN_APP
    )
    assert in_app_status_changed.enabled is True  # in_app is in the default channel set


async def test_mark_read_and_mark_all_read() -> None:
    user_id = uuid.uuid4()

    async with AsyncSessionLocal() as session:
        service = _build_service(session)
        notifications = await service.notify(
            user_id=user_id,
            event_type=NotificationEventType.EMAIL_REVIEW_NEEDED,
            title="Review needed",
            body="Body",
            recipient_email=None,
        )
        await session.commit()

    in_app = next(n for n in notifications if n.channel == NotificationChannel.IN_APP)

    async with AsyncSessionLocal() as session:
        service = _build_service(session)
        read = await service.mark_read(user_id=user_id, notification_id=in_app.id)
        await session.commit()
        assert read.read_at is not None

    async with AsyncSessionLocal() as session:
        service = _build_service(session)
        marked = await service.mark_all_read(user_id=user_id)
        await session.commit()
        assert marked == 0  # already read above
