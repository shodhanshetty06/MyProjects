"""Notification module — data access layer."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.notification.models import (
    Notification,
    NotificationChannel,
    NotificationEventType,
    NotificationPreference,
)
from app.repositories.base import BaseRepository


class NotificationRepository(BaseRepository[Notification]):
    model = Notification

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    def _apply_filters(self, stmt, *, user_id: uuid.UUID, unread_only: bool):
        stmt = stmt.where(Notification.user_id == user_id)
        if unread_only:
            stmt = stmt.where(Notification.read_at.is_(None))
        return stmt

    async def list_filtered(
        self, *, user_id: uuid.UUID, unread_only: bool, offset: int, limit: int
    ) -> Sequence[Notification]:
        stmt = self._apply_filters(select(Notification), user_id=user_id, unread_only=unread_only)
        stmt = stmt.order_by(Notification.created_at.desc()).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def count_filtered(self, *, user_id: uuid.UUID, unread_only: bool) -> int:
        stmt = self._apply_filters(
            select(func.count(Notification.id)), user_id=user_id, unread_only=unread_only
        )
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def get_owned(
        self, *, notification_id: uuid.UUID, user_id: uuid.UUID
    ) -> Notification | None:
        stmt = select(Notification).where(
            Notification.id == notification_id, Notification.user_id == user_id
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def mark_all_read(self, *, user_id: uuid.UUID) -> int:
        stmt = select(Notification).where(
            Notification.user_id == user_id, Notification.read_at.is_(None)
        )
        result = await self.session.execute(stmt)
        notifications = result.scalars().all()
        now = datetime.now(UTC)
        for notification in notifications:
            notification.read_at = now
        await self.session.flush()
        return len(notifications)


class NotificationPreferenceRepository(BaseRepository[NotificationPreference]):
    model = NotificationPreference

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def list_by_user(self, *, user_id: uuid.UUID) -> Sequence[NotificationPreference]:
        stmt = select(NotificationPreference).where(NotificationPreference.user_id == user_id)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_for_user_event_channel(
        self, *, user_id: uuid.UUID, event_type: NotificationEventType, channel: NotificationChannel
    ) -> NotificationPreference | None:
        stmt = select(NotificationPreference).where(
            NotificationPreference.user_id == user_id,
            NotificationPreference.event_type == event_type,
            NotificationPreference.channel == channel,
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def upsert(
        self,
        *,
        user_id: uuid.UUID,
        event_type: NotificationEventType,
        channel: NotificationChannel,
        enabled: bool,
    ) -> NotificationPreference:
        existing = await self.get_for_user_event_channel(
            user_id=user_id, event_type=event_type, channel=channel
        )
        if existing is not None:
            existing.enabled = enabled
            await self.session.flush()
            return existing
        return await self.create(
            user_id=user_id, event_type=event_type, channel=channel, enabled=enabled
        )


__all__ = ["NotificationRepository", "NotificationPreferenceRepository"]
