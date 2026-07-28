"""Voyage AI embedding provider — ported from the pre-refactor `AIService`.
Ready to select via `EMBEDDING_PROVIDER=voyage` once `VOYAGE_API_KEY` is set;
no other application code changes needed.
"""

from __future__ import annotations

import asyncio

from voyageai import Client as VoyageClient

from app.core.config import Settings
from app.core.exceptions import TransientProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)

_VOYAGE_DIMENSIONS = 1024


class VoyageEmbeddingProvider:
    """Text embeddings via the Voyage AI SDK (synchronous — offloaded to a thread).

    The SDK client is constructed lazily (on first use, not in `__init__`) —
    `voyageai.Client` raises `AuthenticationError` immediately if no API key
    is configured, so eager construction would break, e.g., a factory-level
    test that constructs every provider class without every provider's key set.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client_instance: VoyageClient | None = None

    @property
    def _client(self) -> VoyageClient:
        if self._client_instance is None:
            self._client_instance = VoyageClient(api_key=self._settings.VOYAGE_API_KEY)
        return self._client_instance

    @property
    def dimensions(self) -> int:
        return _VOYAGE_DIMENSIONS

    @property
    def model_name(self) -> str:
        return self._settings.VOYAGE_MODEL

    async def embed(self, *, text: str, input_type: str = "document") -> list[float]:
        try:
            result = await asyncio.to_thread(
                self._client.embed,
                [text],
                model=self._settings.VOYAGE_MODEL,
                input_type=input_type,
                output_dimension=_VOYAGE_DIMENSIONS,
            )
        except Exception as exc:
            logger.error(
                "Voyage embedding request failed",
                extra={"extra_fields": {"error": str(exc)}},
            )
            raise TransientProviderError("Voyage AI is unavailable.") from exc
        return result.embeddings[0]


__all__ = ["VoyageEmbeddingProvider"]
