"""Thin JSON cache wrapper around the shared Redis client.

Generic, reusable infra (not module-specific) — any module can inject a
`CacheClient` via `app.api.deps.CacheDep` to add cache-aside behavior around
expensive read paths, following the same "wrap the 3rd-party client behind
our own thin class" convention used by `app.providers.storage.s3_provider.S3StorageProvider`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from redis.asyncio import Redis


class CacheClient:
    """Cache-aside helper: JSON-serialized GET/SETEX over a Redis client."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def get_json(self, key: str) -> Any | None:
        raw = await self._redis.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None

    async def set_json(self, key: str, value: Any, *, ttl_seconds: int) -> None:
        await self._redis.set(key, json.dumps(value, default=str), ex=ttl_seconds)

    async def delete(self, *keys: str) -> None:
        if keys:
            await self._redis.delete(*keys)

    @staticmethod
    def build_key(namespace: str, **params: Any) -> str:
        """Deterministic cache key: same params (any order) -> same key."""
        payload = json.dumps(params, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return f"{namespace}:{digest}"
