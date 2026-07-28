"""Google Gemini embedding provider. Ready to select via
`EMBEDDING_PROVIDER=gemini` once `GEMINI_API_KEY` is set; no other
application code changes needed.
"""

from __future__ import annotations

import google.generativeai as genai

from app.core.config import Settings
from app.core.exceptions import TransientProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)

_GEMINI_DIMENSIONS = 768  # text-embedding-004 default output width


class GeminiEmbeddingProvider:
    """Text embeddings via the Gemini Embeddings API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        genai.configure(api_key=settings.GEMINI_API_KEY)

    @property
    def dimensions(self) -> int:
        return _GEMINI_DIMENSIONS

    @property
    def model_name(self) -> str:
        return self._settings.GEMINI_EMBEDDING_MODEL

    async def embed(self, *, text: str, input_type: str = "document") -> list[float]:
        task_type = "retrieval_query" if input_type == "query" else "retrieval_document"
        try:
            result = await genai.embed_content_async(
                model=f"models/{self._settings.GEMINI_EMBEDDING_MODEL}",
                content=text,
                task_type=task_type,
            )
        except Exception as exc:
            logger.error(
                "Gemini embedding request failed",
                extra={"extra_fields": {"error": str(exc)}},
            )
            raise TransientProviderError("Gemini is unavailable.") from exc
        return result["embedding"]


__all__ = ["GeminiEmbeddingProvider"]
