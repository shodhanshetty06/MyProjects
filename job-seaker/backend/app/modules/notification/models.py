"""Notification module — ORM models.

Domain shape:
    Notification           — one row per (user, event, channel) delivery —
                              the DB-owning record behind both the in-app
                              notification feed and the audit trail for
                              email/push sends.
    NotificationPreference — per-user, per-event-type, per-channel opt-in.
                              Absence of a row for a given
                              (user, event_type, channel) means "use the
                              platform default" (`NOTIFICATION_DEFAULT_CHANNELS`
                              in `Settings`), not "disabled".

`user_id` has no foreign key: same no-`User`-table convention as
`tracking.Application.user_id` — see `app/api/deps.py`.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base_class import Base, TimestampMixin, UUIDPrimaryKeyMixin


class NotificationEventType(str, enum.Enum):
    APPLICATION_STATUS_CHANGED = "application_status_changed"
    EMAIL_REVIEW_NEEDED = "email_review_needed"
    GMAIL_REAUTH_REQUIRED = "gmail_reauth_required"
    RESUME_READY = "resume_ready"
    NEW_MATCHING_JOB = "new_matching_job"


class NotificationChannel(str, enum.Enum):
    EMAIL = "email"
    PUSH = "push"
    IN_APP = "in_app"


class NotificationStatus(str, enum.Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"


class Notification(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "notification"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    event_type: Mapped[NotificationEventType] = mapped_column(String(50), nullable=False)
    channel: Mapped[NotificationChannel] = mapped_column(String(20), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    recipient_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    data: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    status: Mapped[NotificationStatus] = mapped_column(
        String(20), nullable=False, default=NotificationStatus.PENDING, index=True
    )
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class NotificationPreference(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "notification_preference"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "event_type", "channel", name="uq_notification_preference_user_event_channel"
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    event_type: Mapped[NotificationEventType] = mapped_column(String(50), nullable=False)
    channel: Mapped[NotificationChannel] = mapped_column(String(20), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


__all__ = [
    "NotificationEventType",
    "NotificationChannel",
    "NotificationStatus",
    "Notification",
    "NotificationPreference",
]
