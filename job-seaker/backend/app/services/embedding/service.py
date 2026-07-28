"""Embedding service — the single integration point every module uses for
text embeddings (resume/job embeddings, semantic search, recommendations).
Never imports a provider SDK directly; wraps whatever `EmbeddingProvider`
its constructor was given (see `app/factories/embedding_provider_factory.py`),
adds the same retry behavior as `AIService`, and caches results in Redis so
re-embedding identical text (e.g. re-running a pipeline step) is a cache hit.
"""

from __future__ import annotations

from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import Settings
from app.core.exceptions import TransientProviderError
from app.core.logging import get_logger
from app.integrations.cache import CacheClient
from app.interfaces.embedding_provider import EmbeddingProvider

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(
        self,
        provider: EmbeddingProvider,
        settings: Settings,
        cache: CacheClient | None = None,
    ) -> None:
        self._provider = provider
        self._cache = cache
        self._cache_ttl = settings.EMBEDDING_CACHE_TTL_SECONDS
        self._max_retries = settings.AI_PROVIDER_MAX_RETRIES
        self._backoff_seconds = settings.AI_PROVIDER_RETRY_BACKOFF_SECONDS

    @property
    def dimensions(self) -> int:
        return self._provider.dimensions

    @property
    def model_name(self) -> str:
        return self._provider.model_name

    def _new_retryer(self) -> AsyncRetrying:
        return AsyncRetrying(
            retry=retry_if_exception_type(TransientProviderError),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=self._backoff_seconds),
            reraise=True,
        )

    async def generate_embedding(self, *, text: str, input_type: str = "document") -> list[float]:
        cache_key = None
        if self._cache is not None:
            cache_key = CacheClient.build_key(
                "embedding",
                provider=self._provider.model_name,
                input_type=input_type,
                text=text,
            )
            cached = await self._cache.get_json(cache_key)
            if cached is not None:
                return cached

        async for attempt in self._new_retryer():
            with attempt:
                vector = await self._provider.embed(text=text, input_type=input_type)

        if self._cache is not None and cache_key is not None:
            await self._cache.set_json(cache_key, vector, ttl_seconds=self._cache_ttl)
        return vector


__all__ = ["EmbeddingService"]
