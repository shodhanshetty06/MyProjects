"""Picks the concrete `EmbeddingProvider` implementation from
`settings.EMBEDDING_PROVIDER`.

The single place that knows about every embedding provider's constructor —
nothing outside this module (and `dependency_injection/providers.py`, which
calls it) should import a `providers/embedding/*` class directly.
"""

from __future__ import annotations

from app.core.config import Settings
from app.interfaces.embedding_provider import EmbeddingProvider


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "local":
        from app.providers.embedding.local_provider import LocalEmbeddingProvider

        return LocalEmbeddingProvider(settings)
    if provider == "voyage":
        from app.providers.embedding.voyage_provider import VoyageEmbeddingProvider

        return VoyageEmbeddingProvider(settings)
    if provider == "openai":
        from app.providers.embedding.openai_provider import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(settings)
    if provider == "gemini":
        from app.providers.embedding.gemini_provider import GeminiEmbeddingProvider

        return GeminiEmbeddingProvider(settings)

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER '{settings.EMBEDDING_PROVIDER}'. "
        "Expected one of: local, voyage, openai, gemini."
    )


__all__ = ["build_embedding_provider"]
