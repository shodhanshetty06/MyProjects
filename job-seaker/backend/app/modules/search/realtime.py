"""Realtime push for the Search module — Redis pub/sub bridge.

Mirrors `app.modules.matching.realtime` exactly: new-job matching runs on
Celery workers (a separate process from the API), so the in-process
`app.events.bus.EventBus` can't carry notifications to a connected browser
by itself. Redis (already the Celery broker/cache backend) is reused as the
transport: workers/services `PUBLISH` here, and `workspace.router`'s
`/workspace/ws` WebSocket endpoint additionally subscribes to this channel
(alongside `workspace:updates:*`/`matching:updates:*`) and forwards — no
new WebSocket endpoint is created for this (see plan section 7).

Event types published on this channel: `new_matching_jobs`, `feed_stale`,
`job_saved`, `job_unsaved`.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from app.integrations.redis_client import get_redis_client


def channel_name(user_id: uuid.UUID | str) -> str:
    return f"search:updates:{user_id}"


async def publish_search_event(user_id: uuid.UUID | str, event: dict[str, Any]) -> None:
    redis = get_redis_client()
    await redis.publish(channel_name(user_id), json.dumps(event, default=str))
