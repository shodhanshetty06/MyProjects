"""Notification module — Pydantic request/response contracts."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from app.modules.notification.models import (
    NotificationChannel,
    NotificationEventType,
    NotificationStatus,
)


class NotificationRead(BaseModel):
    id: uuid.UUID
    event_type: NotificationEventType
    channel: NotificationChannel
    title: str
    body: str
    data: dict[str, Any]
    status: NotificationStatus
    read_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class NotificationPreferenceRead(BaseModel):
    event_type: NotificationEventType
    channel: NotificationChannel
    enabled: bool

    model_config = {"from_attributes": True}


class NotificationPreferenceUpdate(BaseModel):
    event_type: NotificationEventType
    channel: NotificationChannel
    enabled: bool


class NotificationPreferenceUpdateRequest(BaseModel):
    preferences: list[NotificationPreferenceUpdate]


__all__ = [
    "NotificationRead",
    "NotificationPreferenceRead",
    "NotificationPreferenceUpdate",
    "NotificationPreferenceUpdateRequest",
]
