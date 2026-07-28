"""Groq AI provider (OpenAI-compatible chat completions API, LPU-hosted open
models). Ready to select via `AI_PROVIDER=groq` once `GROQ_API_KEY` is set;
no other application code changes needed.
"""

from __future__ import annotations

import json

from groq import AsyncGroq

from app.core.config import Settings
from app.core.exceptions import ProviderUnavailableError, TransientProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)


class GroqProvider:
    """Chat/structured-extraction over the Groq Chat Completions API.

    The SDK client is constructed lazily (on first use, not in `__init__`) —
    `AsyncGroq` raises immediately if no API key is configured; see
    `AnthropicProvider` for the full rationale.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client_instance: AsyncGroq | None = None

    @property
    def _client(self) -> AsyncGroq:
        if self._client_instance is None:
            self._client_instance = AsyncGroq(api_key=self._settings.GROQ_API_KEY)
        return self._client_instance

    @property
    def model_name(self) -> str:
        return self._settings.GROQ_MODEL

    async def extract_structured(
        self, *, text: str, schema: dict, system: str, max_tokens: int = 8000
    ) -> dict:
        # Groq's JSON mode guarantees valid JSON but not schema conformance,
        # so the schema is embedded in the system prompt as guidance.
        schema_system = f"{system}\n\nRespond with JSON matching this schema:\n{json.dumps(schema)}"
        try:
            response = await self._client.chat.completions.create(
                model=self._settings.GROQ_MODEL,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": schema_system},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.error(
                "Groq structured extraction request failed",
                extra={"extra_fields": {"error": str(exc)}},
            )
            raise TransientProviderError("Groq is unavailable.") from exc

        choice = response.choices[0] if response.choices else None
        if choice is None or choice.message.content is None:
            raise ProviderUnavailableError("Groq returned no content.")
        try:
            return json.loads(choice.message.content)
        except (TypeError, ValueError) as exc:
            raise ProviderUnavailableError("Groq returned invalid JSON.") from exc

    async def complete_chat(
        self, *, system: str, messages: list[dict[str, str]], max_tokens: int = 1500
    ) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._settings.GROQ_MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, *messages],
            )
        except Exception as exc:
            logger.error(
                "Groq chat completion request failed",
                extra={"extra_fields": {"error": str(exc)}},
            )
            raise TransientProviderError("Groq is unavailable.") from exc

        choice = response.choices[0] if response.choices else None
        if choice is None or choice.message.content is None:
            raise ProviderUnavailableError("Groq returned no content.")
        return choice.message.content


__all__ = ["GroqProvider"]
