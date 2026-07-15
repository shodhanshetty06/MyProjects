"""Redis connection management.

A single async connection pool is created lazily and reused for the life
of the process. Used directly for caching and also as the backing broker
for Celery.
"""
from __future__ import annotations

from functools import lru_cache

from redis.asyncio import Redis, from_url

from app.core.config import get_settings


@lru_cache
def get_redis_client() -> Redis:
    settings = get_settings()
    return from_url(settings.REDIS_URL, decode_responses=True)


async def close_redis_client() -> None:
    client = get_redis_client()
    await client.aclose()
