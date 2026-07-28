"""Unit tests for `CacheClient` — JSON round-trip, TTL pass-through, and
deterministic key building. Uses a hand-rolled fake Redis (no real network),
matching the fake-provider-client convention used elsewhere in the test
suite (e.g. `_FakeStorageProvider`, `_FakeEmbeddingProvider`).
"""

from __future__ import annotations

import pytest

from app.integrations.cache import CacheClient


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.set_calls: list[dict] = []

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, *, ex: int | None = None) -> None:
        self.store[key] = value
        self.set_calls.append({"key": key, "value": value, "ex": ex})

    async def delete(self, *keys: str) -> None:
        for key in keys:
            self.store.pop(key, None)


@pytest.mark.asyncio
async def test_get_json_returns_none_on_miss() -> None:
    cache = CacheClient(_FakeRedis())
    assert await cache.get_json("missing") is None


@pytest.mark.asyncio
async def test_set_json_then_get_json_round_trips() -> None:
    cache = CacheClient(_FakeRedis())
    await cache.set_json("key", {"a": 1, "b": [1, 2, 3]}, ttl_seconds=60)
    assert await cache.get_json("key") == {"a": 1, "b": [1, 2, 3]}


@pytest.mark.asyncio
async def test_set_json_passes_through_ttl() -> None:
    redis = _FakeRedis()
    cache = CacheClient(redis)
    await cache.set_json("key", {"a": 1}, ttl_seconds=120)
    assert redis.set_calls[0]["ex"] == 120


@pytest.mark.asyncio
async def test_delete_removes_key() -> None:
    redis = _FakeRedis()
    cache = CacheClient(redis)
    await cache.set_json("key", {"a": 1}, ttl_seconds=60)
    await cache.delete("key")
    assert await cache.get_json("key") is None


def test_build_key_is_deterministic_regardless_of_param_order() -> None:
    key_a = CacheClient.build_key("ns", a=1, b=2)
    key_b = CacheClient.build_key("ns", b=2, a=1)
    assert key_a == key_b


def test_build_key_differs_for_different_params() -> None:
    key_a = CacheClient.build_key("ns", a=1)
    key_b = CacheClient.build_key("ns", a=2)
    assert key_a != key_b


def test_build_key_includes_namespace_prefix() -> None:
    key = CacheClient.build_key("search:v1:query", a=1)
    assert key.startswith("search:v1:query:")
