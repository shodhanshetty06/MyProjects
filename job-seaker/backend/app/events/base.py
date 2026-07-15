"""Base event contract for the in-process event bus."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class Event(BaseModel):
    """Base class every domain event should subclass.

    Example (future phase):
        class UserRegisteredEvent(Event):
            name: str = "user.registered"
            user_id: uuid.UUID
    """

    event_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = Field(default_factory=dict)
