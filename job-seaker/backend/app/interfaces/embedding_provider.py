"""Contract every embedding provider (local sentence-transformers, Voyage,
OpenAI, Gemini) must satisfy so `EmbeddingService` never has to know which
one is behind it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """A text-embedding backend."""

    @property
    def dimensions(self) -> int:
        """Output vector width — must match `settings.VECTOR_DIMENSIONS`."""
        ...

    @property
    def model_name(self) -> str:
        """Model identifier, used in embedding-cache keys and `model_used` columns."""
        ...

    async def embed(self, *, text: str, input_type: str = "document") -> list[float]:
        """Generate a vector embedding for `text`.

        `input_type` is `"document"` for content being indexed (resumes, job
        postings) and `"query"` for a search query — providers that support
        asymmetric embeddings (e.g. Voyage) use it to improve retrieval
        quality; providers that don't may ignore it.
        """
        ...
