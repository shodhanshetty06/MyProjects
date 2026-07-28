"""Google Gemini AI provider. Ready to select via `AI_PROVIDER=gemini` once
`GEMINI_API_KEY` is set; no other application code changes needed.
"""

from __future__ import annotations

import json

import google.generativeai as genai

from app.core.config import Settings
from app.core.exceptions import ProviderUnavailableError, TransientProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)


class GeminiProvider:
    """Chat/structured-extraction over the Gemini API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        genai.configure(api_key=settings.GEMINI_API_KEY)

    @property
    def model_name(self) -> str:
        return self._settings.GEMINI_MODEL

    async def extract_structured(
        self, *, text: str, schema: dict, system: str, max_tokens: int = 8000
    ) -> dict:
        model = genai.GenerativeModel(self._settings.GEMINI_MODEL, system_instruction=system)
        try:
            response = await model.generate_content_async(
                text,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
        except Exception as exc:
            logger.error(
                "Gemini structured extraction request failed",
                extra={"extra_fields": {"error": str(exc)}},
            )
            raise TransientProviderError("Gemini is unavailable.") from exc

        if not response.candidates or not response.text:
            raise ProviderUnavailableError("Gemini returned no content.")
        try:
            return json.loads(response.text)
        except (TypeError, ValueError) as exc:
            raise ProviderUnavailableError("Gemini returned invalid JSON.") from exc

    async def complete_chat(
        self, *, system: str, messages: list[dict[str, str]], max_tokens: int = 1500
    ) -> str:
        model = genai.GenerativeModel(self._settings.GEMINI_MODEL, system_instruction=system)
        history = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [m["content"]]}
            for m in messages[:-1]
        ]
        try:
            chat = model.start_chat(history=history)
            response = await chat.send_message_async(
                messages[-1]["content"],
                generation_config=genai.GenerationConfig(max_output_tokens=max_tokens),
            )
        except Exception as exc:
            logger.error(
                "Gemini chat completion request failed",
                extra={"extra_fields": {"error": str(exc)}},
            )
            raise TransientProviderError("Gemini is unavailable.") from exc

        if not response.candidates or not response.text:
            raise ProviderUnavailableError("Gemini returned no content.")
        return response.text


__all__ = ["GeminiProvider"]
