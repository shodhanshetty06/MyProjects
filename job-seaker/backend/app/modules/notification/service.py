"""Notification module — application/use-case layer.

The single `notify(...)` entrypoint every other module (currently
`email_tracking`) calls into — mirrors the `app/services/ai` (cross-cutting
provider) vs. `app/modules/resume_parser` (DB + routes) split: this module
owns persistence, preference resolution, and ownership checks, and delegates
actual delivery to the pure channel providers in `app/services/notification/`.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from app.core.config import get_settings
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.modules.notification.models import (
    Notification,
    NotificationChannel,
    NotificationEventType,
    NotificationStatus,
)
from app.modules.notification.repository import (
    NotificationPreferenceRepository,
    NotificationRepository,
)
from app.modules.notification.schemas import NotificationPreferenceUpdate
from app.services.notification.push_sender import PushSender

logger = get_logger(__name__)


@dataclass
class EffectivePreference:
    event_type: NotificationEventType
    channel: NotificationChannel
    enabled: bool


class NotificationService:
    def __init__(
        self,
        *,
        repository: NotificationRepository,
        preference_repository: NotificationPreferenceRepository,
        push_sender: PushSender | None = None,
    ) -> None:
        self.repository = repository
        self.preference_repository = preference_repository
        self.push_sender = push_sender or PushSender()
        self.settings = get_settings()

    def _default_channels(self) -> set[NotificationChannel]:
        return {NotificationChannel(value) for value in self.settings.notification_default_channels}

    async def _resolve_enabled_channels(
        self, *, user_id: uuid.UUID, event_type: NotificationEventType
    ) -> list[NotificationChannel]:
        preferences = await self.preference_repository.list_by_user(user_id=user_id)
        overrides = {p.channel: p.enabled for p in preferences if p.event_type == event_type}
        defaults = self._default_channels()
        return [
            channel
            for channel in NotificationChannel
            if overrides.get(channel, channel in defaults)
        ]

    async def notify(
        self,
        *,
        user_id: uuid.UUID,
        event_type: NotificationEventType,
        title: str,
        body: str,
        data: dict | None = None,
        recipient_email: str | None = None,
    ) -> list[Notification]:
        """Resolves enabled channels for `(user_id, event_type)` and dispatches
        to each. IN_APP and PUSH are handled synchronously (PUSH is a logged
        placeholder — see `app/services/notification/push_sender.py`). EMAIL
        writes a PENDING row and enqueues `notification.send_email` so a
        slow/throttled SES call never blocks the caller. If EMAIL resolves as
        enabled but `recipient_email` is absent (there is no `User` table to
        look an address up from — see the module's design notes), that leg is
        skipped with a warning; other channels still proceed.
        """
        from app.modules.notification.tasks import send_notification_email_task

        data = data or {}
        channels = await self._resolve_enabled_channels(user_id=user_id, event_type=event_type)
        notifications: list[Notification] = []

        for channel in channels:
            if channel == NotificationChannel.EMAIL:
                if not recipient_email:
                    logger.warning(
                        "Skipping EMAIL notification — no recipient_email supplied",
                        extra={
                            "extra_fields": {
                                "user_id": str(user_id),
                                "event_type": event_type.value,
                            }
                        },
                    )
                    continue
                notification = await self.repository.create(
                    user_id=user_id,
                    event_type=event_type,
                    channel=channel,
                    title=title,
                    body=body,
                    recipient_email=recipient_email,
                    data=data,
                    status=NotificationStatus.PENDING,
                )
                await self.repository.session.flush()
                send_notification_email_task.delay(str(notification.id))
                notifications.append(notification)

            elif channel == NotificationChannel.IN_APP:
                notification = await self.repository.create(
                    user_id=user_id,
                    event_type=event_type,
                    channel=channel,
                    title=title,
                    body=body,
                    data=data,
                    status=NotificationStatus.SENT,
                )
                notifications.append(notification)

            elif channel == NotificationChannel.PUSH:
                await self.push_sender.send(user_id=str(user_id), title=title, body=body)
                notification = await self.repository.create(
                    user_id=user_id,
                    event_type=event_type,
                    channel=channel,
                    title=title,
                    body=body,
                    data=data,
                    status=NotificationStatus.SENT,
                )
                notifications.append(notification)

        return notifications

    async def list_notifications(
        self, *, user_id: uuid.UUID, unread_only: bool, offset: int, limit: int
    ) -> tuple[Sequence[Notification], int]:
        notifications = await self.repository.list_filtered(
            user_id=user_id, unread_only=unread_only, offset=offset, limit=limit
        )
        total = await self.repository.count_filtered(user_id=user_id, unread_only=unread_only)
        return notifications, total

    async def mark_read(self, *, user_id: uuid.UUID, notification_id: uuid.UUID) -> Notification:
        notification = await self.repository.get_owned(
            notification_id=notification_id, user_id=user_id
        )
        if notification is None:
            raise NotFoundError("Notification not found.")
        return await self.repository.update(notification, read_at=datetime.now(UTC))

    async def mark_all_read(self, *, user_id: uuid.UUID) -> int:
        return await self.repository.mark_all_read(user_id=user_id)

    async def get_preferences(self, *, user_id: uuid.UUID) -> list[EffectivePreference]:
        existing = await self.preference_repository.list_by_user(user_id=user_id)
        overrides = {(p.event_type, p.channel): p.enabled for p in existing}
        defaults = self._default_channels()
        return [
            EffectivePreference(
                event_type=event_type,
                channel=channel,
                enabled=overrides.get((event_type, channel), channel in defaults),
            )
            for event_type in NotificationEventType
            for channel in NotificationChannel
        ]

    async def update_preferences(
        self, *, user_id: uuid.UUID, updates: list[NotificationPreferenceUpdate]
    ) -> list[EffectivePreference]:
        for update in updates:
            await self.preference_repository.upsert(
                user_id=user_id,
                event_type=update.event_type,
                channel=update.channel,
                enabled=update.enabled,
            )
        return await self.get_preferences(user_id=user_id)


__all__ = ["NotificationService", "EffectivePreference"]
