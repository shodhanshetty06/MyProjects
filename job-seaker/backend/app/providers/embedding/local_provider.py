"""Local sentence-transformers embedding provider — the primary/default
`EmbeddingProvider`. Runs fully offline (after the model's first download
from HuggingFace Hub) and needs no API key.

The model is loaded lazily (on first `.embed()`/`.dimensions` access, not in
`__init__`) — same convention `AIService`'s Anthropic/Voyage clients used to
follow: constructing a `LocalEmbeddingProvider` (e.g. as a FastAPI request
dependency) shouldn't pay the cost of loading model weights unless the
request actually generates an embedding. The loaded model is cached once
per process (`@lru_cache`, keyed by model name), so concurrent/repeated
`LocalEmbeddingProvider` instances for the same model reuse it rather than
reloading.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import Settings
from app.core.exceptions import ProviderUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache
def _get_model(model_name: str) -> SentenceTransformer:
    logger.info("Loading local embedding model", extra={"extra_fields": {"model": model_name}})
    return SentenceTransformer(model_name)


class LocalEmbeddingProvider:
    """Text embeddings via a local sentence-transformers model."""

    def __init__(self, settings: Settings) -> None:
        self._model_name_ = settings.LOCAL_EMBEDDING_MODEL
        self._model_instance: SentenceTransformer | None = None

    @property
    def _model(self) -> SentenceTransformer:
        if self._model_instance is None:
            self._model_instance = _get_model(self._model_name_)
        return self._model_instance

    @property
    def dimensions(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name_

    async def embed(self, *, text: str, input_type: str = "document") -> list[float]:
        try:
            vector = await asyncio.to_thread(self._model.encode, text, normalize_embeddings=True)
        except Exception as exc:
            logger.error(
                "Local embedding generation failed",
                extra={"extra_fields": {"model": self._model_name_, "error": str(exc)}},
            )
            raise ProviderUnavailableError("Local embedding provider failed.") from exc
        return vector.tolist()


__all__ = ["LocalEmbeddingProvider"]
