"""OpenAI AI provider. Ready to select via `AI_PROVIDER=openai` once
`OPENAI_API_KEY` is set; no other application code changes needed.
"""

from __future__ import annotations

import json

from openai import AsyncOpenAI

from app.core.config import Settings
from app.core.exceptions import ProviderUnavailableError, TransientProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)


class OpenAIProvider:
    """Chat/structured-extraction over the OpenAI Chat Completions API.

    The SDK client is constructed lazily (on first use, not in `__init__`) —
    `AsyncOpenAI` raises immediately if no API key is configured; see
    `AnthropicProvider` for the full rationale.
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
    def model_name(self) -> str:
        return self._settings.OPENAI_MODEL

    async def extract_structured(
        self, *, text: str, schema: dict, system: str, max_tokens: int = 8000
    ) -> dict:
        try:
            response = await self._client.chat.completions.create(
                model=self._settings.OPENAI_MODEL,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": text},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "extraction", "schema": schema, "strict": True},
                },
            )
        except Exception as exc:
            logger.error(
                "OpenAI structured extraction request failed",
                extra={"extra_fields": {"error": str(exc)}},
            )
            raise TransientProviderError("OpenAI is unavailable.") from exc

        choice = response.choices[0] if response.choices else None
        if choice is None or choice.message.content is None:
            raise ProviderUnavailableError("OpenAI returned no content.")
        if choice.finish_reason == "content_filter":
            raise ProviderUnavailableError("Extraction was refused by the model.")
        try:
            return json.loads(choice.message.content)
        except (TypeError, ValueError) as exc:
            raise ProviderUnavailableError("OpenAI returned invalid JSON.") from exc

    async def complete_chat(
        self, *, system: str, messages: list[dict[str, str]], max_tokens: int = 1500
    ) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._settings.OPENAI_MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, *messages],
            )
        except Exception as exc:
            logger.error(
                "OpenAI chat completion request failed",
                extra={"extra_fields": {"error": str(exc)}},
            )
            raise TransientProviderError("OpenAI is unavailable.") from exc

        choice = response.choices[0] if response.choices else None
        if choice is None or choice.message.content is None:
            raise ProviderUnavailableError("OpenAI returned no content.")
        if choice.finish_reason == "content_filter":
            raise ProviderUnavailableError("Response was refused by the model.")
        return choice.message.content


__all__ = ["OpenAIProvider"]
