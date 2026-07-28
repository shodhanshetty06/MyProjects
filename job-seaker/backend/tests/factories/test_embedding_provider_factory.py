"""Provider-switching tests for `build_embedding_provider` — verifies
`EMBEDDING_PROVIDER` alone selects the right concrete class. Local provider
construction no longer eagerly loads model weights (lazy on first
`.embed()`/`.dimensions` access), so this needs no network access.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.factories.embedding_provider_factory import build_embedding_provider
from app.providers.embedding.gemini_provider import GeminiEmbeddingProvider
from app.providers.embedding.local_provider import LocalEmbeddingProvider
from app.providers.embedding.openai_provider import OpenAIEmbeddingProvider
from app.providers.embedding.voyage_provider import VoyageEmbeddingProvider


@pytest.mark.parametrize(
    "provider_name,expected_class",
    [
        ("local", LocalEmbeddingProvider),
        ("voyage", VoyageEmbeddingProvider),
        ("openai", OpenAIEmbeddingProvider),
        ("gemini", GeminiEmbeddingProvider),
    ],
)
def test_build_embedding_provider_selects_class_from_setting(provider_name, expected_class) -> None:
    settings = Settings(EMBEDDING_PROVIDER=provider_name)

    provider = build_embedding_provider(settings)

    assert isinstance(provider, expected_class)


def test_build_embedding_provider_rejects_unknown_provider() -> None:
    settings = Settings(EMBEDDING_PROVIDER="not-a-real-provider")

    with pytest.raises(ValueError, match="Unknown EMBEDDING_PROVIDER"):
        build_embedding_provider(settings)
