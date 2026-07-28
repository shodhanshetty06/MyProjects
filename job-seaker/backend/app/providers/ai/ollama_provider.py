"""Ollama AI provider — the primary/default `AIProvider`, talking to a local
(or self-hosted) Ollama server over HTTP. No API key required.

Structured extraction uses Ollama's `format` chat parameter with a JSON
schema (the same idea as Anthropic's `output_config`); free-form chat omits
it. If the configured model isn't pulled yet, each entry in
`OLLAMA_FALLBACK_MODELS` is tried in order before giving up.
"""

from __future__ import annotations

import json

import httpx
import ollama

from app.core.config import Settings
from app.core.exceptions import ProviderUnavailableError, TransientProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)


def _is_model_missing(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    return status_code == 404 or "not found" in str(exc).lower()


def _is_transient(exc: Exception) -> bool:
    return isinstance(exc, (httpx.ConnectError, httpx.TimeoutException, httpx.ConnectTimeout))


# NOTE: deliberately NOT caching `ollama.AsyncClient` (or its underlying
# `httpx.AsyncClient`) at module scope across `OllamaProvider` instances.
# Every Celery task in this codebase runs its own `asyncio.run(...)` (see
# `app.workers.celery_app._reset_async_resources`'s docstring, which
# documents the identical constraint for the DB engine and Redis client),
# and httpx's async connection pool is bound to the event loop that created
# it. Reusing one across tasks was tried and empirically reproduces
# `RuntimeError: Event loop is closed` the moment a second task on the same
# worker process tries to use the cached client — the same failure mode the
# existing Redis/engine cache-clearing in `_reset_async_resources` exists to
# avoid. A fresh client per task is therefore correct, not an oversight.
class OllamaProvider:
    """Chat/structured-extraction over the Ollama HTTP API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = ollama.AsyncClient(
            host=settings.OLLAMA_BASE_URL, timeout=settings.OLLAMA_REQUEST_TIMEOUT_SECONDS
        )
        self._models = [settings.OLLAMA_MODEL, *settings.ollama_fallback_models]

    @property
    def model_name(self) -> str:
        return self._settings.OLLAMA_MODEL

    async def _chat(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        format: dict | None = None,
        temperature: float | None = None,
    ):
        options: dict = {"num_predict": max_tokens}
        if temperature is not None:
            options["temperature"] = temperature
        last_exc: Exception | None = None
        for model in self._models:
            try:
                return await self._client.chat(
                    model=model,
                    messages=messages,
                    format=format,
                    options=options,
                )
            except (httpx.ConnectError, httpx.TimeoutException, httpx.ConnectTimeout) as exc:
                logger.error(
                    "Ollama request failed — server unreachable",
                    extra={"extra_fields": {"model": model, "error": str(exc)}},
                )
                raise TransientProviderError("Ollama is offline or unreachable.") from exc
            except ollama.ResponseError as exc:
                if _is_model_missing(exc) and model != self._models[-1]:
                    logger.warning(
                        "Ollama model not installed, trying fallback",
                        extra={"extra_fields": {"model": model, "error": str(exc)}},
                    )
                    last_exc = exc
                    continue
                logger.error(
                    "Ollama request failed",
                    extra={"extra_fields": {"model": model, "error": str(exc)}},
                )
                raise ProviderUnavailableError(
                    f"Ollama model '{model}' is unavailable and no fallback succeeded."
                ) from exc
        raise ProviderUnavailableError("All configured Ollama models are unavailable.") from last_exc

    async def extract_structured(
        self, *, text: str, schema: dict, system: str, max_tokens: int = 8000
    ) -> dict:
        response = await self._chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            max_tokens=max_tokens,
            format=schema,
            # Low temperature for extraction (vs. `complete_chat`'s default):
            # this is a "find what's on the page" task, not a creative one,
            # and a lower temperature measurably reduces run-to-run variance
            # in which fields a small local model bothers to fill in.
            temperature=0.2,
        )
        content = response.message.content
        if not content:
            raise ProviderUnavailableError("Ollama returned no content.")
        try:
            return json.loads(content)
        except (TypeError, ValueError) as exc:
            raise ProviderUnavailableError("Ollama returned invalid JSON.") from exc

    async def complete_chat(
        self, *, system: str, messages: list[dict[str, str]], max_tokens: int = 1500
    ) -> str:
        response = await self._chat(
            messages=[{"role": "system", "content": system}, *messages],
            max_tokens=max_tokens,
        )
        content = response.message.content
        if not content:
            raise ProviderUnavailableError("Ollama returned no content.")
        return content


__all__ = ["OllamaProvider"]
