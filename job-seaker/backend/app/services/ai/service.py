"""AI service — the single integration point for LLM-backed capabilities
used across modules: structured extraction (Claude) and text embeddings
(Voyage AI, since the Claude API has no embeddings endpoint).
"""

from __future__ import annotations

import asyncio
import json

from anthropic import AsyncAnthropic
from voyageai import Client as VoyageClient

from app.core.config import get_settings
from app.core.exceptions import ServiceUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)


class AIService:
    """Coordinates all LLM/embedding provider calls for the platform.

    Provider clients are constructed lazily (on first use, not in
    `__init__`) — `voyageai.Client` raises `AuthenticationError` immediately
    if no API key is configured, and eagerly constructing it here would mean
    a deployment (or test) that only needs Claude extraction couldn't
    instantiate this service without a Voyage key too.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._anthropic_client: AsyncAnthropic | None = None
        self._voyage_client: VoyageClient | None = None

    @property
    def _anthropic(self) -> AsyncAnthropic:
        if self._anthropic_client is None:
            self._anthropic_client = AsyncAnthropic(api_key=self._settings.ANTHROPIC_API_KEY)
        return self._anthropic_client

    @property
    def _voyage(self) -> VoyageClient:
        if self._voyage_client is None:
            self._voyage_client = VoyageClient(api_key=self._settings.VOYAGE_API_KEY)
        return self._voyage_client

    async def generate_embedding(self, *, text: str, input_type: str = "document") -> list[float]:
        """Generate a vector embedding for the given text via Voyage AI.

        `input_type` should be `"document"` when embedding content being
        indexed (resumes, job postings) and `"query"` when embedding a
        search query — Voyage's asymmetric embeddings improve retrieval
        quality when the two sides are marked accordingly.

        The Voyage SDK is synchronous, so the call is offloaded to a thread
        to avoid blocking the event loop.
        """
        try:
            result = await asyncio.to_thread(
                self._voyage.embed,
                [text],
                model=self._settings.VOYAGE_MODEL,
                input_type=input_type,
                output_dimension=self._settings.VECTOR_DIMENSIONS,
            )
        except Exception as exc:
            logger.error(
                "Voyage embedding request failed",
                extra={"extra_fields": {"error": str(exc)}},
            )
            raise ServiceUnavailableError("Embedding provider is unavailable.") from exc
        return result.embeddings[0]

    async def extract_structured(
        self, *, text: str, schema: dict, system: str, max_tokens: int = 8000
    ) -> dict:
        """Extract structured JSON from `text` per the given JSON schema.

        Generic by design (not resume-specific) so other modules can reuse
        it for their own structured-extraction needs.
        """
        try:
            response = await self._anthropic.messages.create(
                model=self._settings.ANTHROPIC_MODEL,
                max_tokens=max_tokens,
                system=system,
                output_config={"format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": text}],
            )
        except Exception as exc:
            logger.error(
                "Claude structured extraction request failed",
                extra={"extra_fields": {"error": str(exc)}},
            )
            raise ServiceUnavailableError("Extraction provider is unavailable.") from exc

        if response.stop_reason == "refusal":
            raise ServiceUnavailableError("Extraction was refused by the model.")

        text_block = next((block for block in response.content if block.type == "text"), None)
        if text_block is None:
            raise ServiceUnavailableError("Extraction provider returned no content.")
        return json.loads(text_block.text)

    async def complete_chat(
        self, *, system: str, messages: list[dict[str, str]], max_tokens: int = 1500
    ) -> str:
        """Generate a free-form conversational reply from Claude.

        Unlike `extract_structured`, the response is plain text rather than
        JSON — for conversational assistants where the reply isn't meant to
        fit a fixed schema. `messages` is oldest-first, each a
        `{"role": "user"|"assistant", "content": str}` dict. Generic by
        design so any module needing free-form chat completion can reuse it.
        """
        try:
            response = await self._anthropic.messages.create(
                model=self._settings.ANTHROPIC_MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
        except Exception as exc:
            logger.error(
                "Claude chat completion request failed",
                extra={"extra_fields": {"error": str(exc)}},
            )
            raise ServiceUnavailableError("Chat completion provider is unavailable.") from exc

        if response.stop_reason == "refusal":
            raise ServiceUnavailableError("Response was refused by the model.")

        text_block = next((block for block in response.content if block.type == "text"), None)
        if text_block is None:
            raise ServiceUnavailableError("Chat completion provider returned no content.")
        return text_block.text
