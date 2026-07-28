"""Anthropic (Claude) AI provider — ported from the pre-refactor `AIService`.
Ready to select via `AI_PROVIDER=anthropic` once `ANTHROPIC_API_KEY` is set;
no other application code changes needed.
"""

from __future__ import annotations

import json

from anthropic import AsyncAnthropic

from app.core.config import Settings
from app.core.exceptions import ProviderUnavailableError, TransientProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)


class AnthropicProvider:
    """Chat/structured-extraction over the Anthropic Messages API.

    The SDK client is constructed lazily (on first use, not in `__init__`) —
    `AsyncAnthropic` raises immediately if no API key is configured, and this
    provider is only built when `AI_PROVIDER=anthropic` is actually selected,
    but eager construction would still break, e.g., a factory-level test
    that constructs every provider class without every provider's key set.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client_instance: AsyncAnthropic | None = None

    @property
    def _client(self) -> AsyncAnthropic:
        if self._client_instance is None:
            self._client_instance = AsyncAnthropic(api_key=self._settings.ANTHROPIC_API_KEY)
        return self._client_instance

    @property
    def model_name(self) -> str:
        return self._settings.ANTHROPIC_MODEL

    async def extract_structured(
        self, *, text: str, schema: dict, system: str, max_tokens: int = 8000
    ) -> dict:
        try:
            response = await self._client.messages.create(
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
            raise TransientProviderError("Anthropic is unavailable.") from exc

        if response.stop_reason == "refusal":
            raise ProviderUnavailableError("Extraction was refused by the model.")

        text_block = next((block for block in response.content if block.type == "text"), None)
        if text_block is None:
            raise ProviderUnavailableError("Anthropic returned no content.")
        return json.loads(text_block.text)

    async def complete_chat(
        self, *, system: str, messages: list[dict[str, str]], max_tokens: int = 1500
    ) -> str:
        try:
            response = await self._client.messages.create(
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
            raise TransientProviderError("Anthropic is unavailable.") from exc

        if response.stop_reason == "refusal":
            raise ProviderUnavailableError("Response was refused by the model.")

        text_block = next((block for block in response.content if block.type == "text"), None)
        if text_block is None:
            raise ProviderUnavailableError("Anthropic returned no content.")
        return text_block.text


__all__ = ["AnthropicProvider"]
