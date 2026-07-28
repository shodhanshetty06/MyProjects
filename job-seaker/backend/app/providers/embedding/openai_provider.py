"""OpenAI embedding provider. Ready to select via
`EMBEDDING_PROVIDER=openai` once `OPENAI_API_KEY` is set; no other
application code changes needed.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from app.core.config import Settings
from app.core.exceptions import TransientProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)

# text-embedding-3-small is 1536-dim by default; text-embedding-3-large is
# 3072. Only -3-small is wired up as the default OPENAI_EMBEDDING_MODEL.
_OPENAI_DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}


class OpenAIEmbeddingProvider:
    """Text embeddings via the OpenAI Embeddings API.

    The SDK client is constructed lazily (on first use, not in `__init__`) —
    `AsyncOpenAI` raises immediately if no API key is configured; see
    `app.providers.ai.anthropic_provider.AnthropicProvider` for the full
    rationale.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client_instance: AsyncOpenAI | None = None

    @property
    def _client(self) -> AsyncOpenAI:
        if self._client_instance is None:
            self._client_instance = AsyncOpenAI(api_key=self._settings.OPENAI_API_KEY)
        return self._client_instance

    @property
    def dimensions(self) -> int:
        return _OPENAI_DIMENSIONS.get(self._settings.OPENAI_EMBEDDING_MODEL, 1536)

    @property
    def model_name(self) -> str:
        return self._settings.OPENAI_EMBEDDING_MODEL

    async def embed(self, *, text: str, input_type: str = "document") -> list[float]:
        try:
            response = await self._client.embeddings.create(
                model=self._settings.OPENAI_EMBEDDING_MODEL, input=text
            )
        except Exception as exc:
            logger.error(
                "OpenAI embedding request failed",
                extra={"extra_fields": {"error": str(exc)}},
            )
            raise TransientProviderError("OpenAI is unavailable.") from exc
        return response.data[0].embedding


__all__ = ["OpenAIEmbeddingProvider"]
