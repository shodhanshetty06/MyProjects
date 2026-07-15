"""Realtime push for the Matching module — Redis pub/sub bridge.

Match/recommendation computation runs on Celery workers (a separate process
from the API), so the in-process `app.events.bus.EventBus` can't carry
notifications to a connected browser by itself. Redis (already the Celery
broker/cache backend) is reused as the transport: workers `PUBLISH` here,
and `matching.router`'s WebSocket endpoint subscribes per-user and forwards.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from app.integrations.redis_client import get_redis_client


def channel_name(user_id: uuid.UUID | str) -> str:
    return f"matching:updates:{user_id}"


async def publish_match_event(user_id: uuid.UUID | str, event: dict[str, Any]) -> None:
    redis = get_redis_client()
    await redis.publish(channel_name(user_id), json.dumps(event, default=str))
