"""Unit tests for `EmbeddingService` — retry behavior (same contract as
`AIService`) plus cache-aside behavior against a fake `CacheClient`-shaped
Redis stand-in. No network access required.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.exceptions import ProviderUnavailableError, TransientProviderError
from app.integrations.cache import CacheClient
from app.services.embedding.service import EmbeddingService


def _settings(**overrides) -> Settings:
    defaults = dict(AI_PROVIDER_MAX_RETRIES=3, AI_PROVIDER_RETRY_BACKOFF_SECONDS=0.001)
    defaults.update(overrides)
    return Settings(**defaults)


class _FakeRedis:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def get(self, key: str):
        return self._store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None):
        self._store[key] = value

    async def delete(self, *keys: str):
        for key in keys:
            self._store.pop(key, None)


class _FakeEmbeddingProvider:
    def __init__(self, *, fail_times: int = 0, vector: list[float] | None = None) -> None:
        self._fail_times = fail_times
        self._vector = vector or [0.1, 0.2, 0.3]
        self.model_name = "fake-embedding-model"
        self.dimensions = len(self._vector)
        self.calls = 0

    async def embed(self, *, text: str, input_type: str = "document") -> list[float]:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise TransientProviderError("offline")
        return self._vector


@pytest.mark.asyncio
async def test_generate_embedding_returns_vector_without_cache() -> None:
    provider = _FakeEmbeddingProvider()
    service = EmbeddingService(provider, _settings())

    result = await service.generate_embedding(text="hello world")

    assert result == [0.1, 0.2, 0.3]
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_generate_embedding_is_cached_on_second_call() -> None:
    provider = _FakeEmbeddingProvider()
    cache = CacheClient(_FakeRedis())
    service = EmbeddingService(provider, _settings(), cache=cache)

    first = await service.generate_embedding(text="hello world")
    second = await service.generate_embedding(text="hello world")

    assert first == second
    assert provider.calls == 1  # second call was served from cache


@pytest.mark.asyncio
async def test_generate_embedding_cache_key_differs_by_input_type() -> None:
    provider = _FakeEmbeddingProvider()
    cache = CacheClient(_FakeRedis())
    service = EmbeddingService(provider, _settings(), cache=cache)

    await service.generate_embedding(text="hello world", input_type="document")
    await service.generate_embedding(text="hello world", input_type="query")

    assert provider.calls == 2  # different input_type -> different cache key


@pytest.mark.asyncio
async def test_generate_embedding_retries_transient_errors() -> None:
    provider = _FakeEmbeddingProvider(fail_times=2)
    service = EmbeddingService(provider, _settings())

    result = await service.generate_embedding(text="hello world")

    assert result == [0.1, 0.2, 0.3]
    assert provider.calls == 3


@pytest.mark.asyncio
async def test_generate_embedding_raises_on_non_retryable_error() -> None:
    class _RaisingProvider:
        model_name = "fake"
        dimensions = 3

        async def embed(self, *, text: str, input_type: str = "document") -> list[float]:
            raise ProviderUnavailableError("bad config")

    service = EmbeddingService(_RaisingProvider(), _settings())

    with pytest.raises(ProviderUnavailableError):
        await service.generate_embedding(text="hello world")
