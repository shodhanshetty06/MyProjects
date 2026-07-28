"""Realtime push for the Workspace module — Redis pub/sub bridge.

Mirrors `app.modules.matching.realtime`: pipeline/dashboard-affecting work
runs on Celery workers (a separate process from the API), so the in-process
`app.events.bus.EventBus` can't carry notifications to a connected browser by
itself. Redis (already the Celery broker/cache backend) is reused as the
transport: workers `PUBLISH` here, and `workspace.router`'s WebSocket
endpoint subscribes per-user (alongside the existing `matching:updates:*`
channel) and forwards.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from app.integrations.redis_client import get_redis_client


def channel_name(user_id: uuid.UUID | str) -> str:
    return f"workspace:updates:{user_id}"


async def publish_workspace_event(user_id: uuid.UUID | str, event: dict[str, Any]) -> None:
    redis = get_redis_client()
    await redis.publish(channel_name(user_id), json.dumps(event, default=str))
