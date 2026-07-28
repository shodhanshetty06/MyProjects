"""AI service — the single integration point every module uses for
LLM-backed capabilities (structured extraction, free-form chat). Never
imports a provider SDK directly; it wraps whatever `AIProvider` its
constructor was given (see `app/factories/ai_provider_factory.py`) and adds
provider-agnostic retry behavior on top.
"""

from __future__ import annotations

from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import Settings
from app.core.exceptions import TransientProviderError
from app.core.logging import get_logger
from app.interfaces.ai_provider import AIProvider

logger = get_logger(__name__)


class AIService:
    """Retries transient provider failures; everything else (bad config,
    model refusal, invalid response) is not retried and propagates as-is —
    retrying those would just waste time reproducing the same failure."""

    def __init__(self, provider: AIProvider, settings: Settings) -> None:
        self._provider = provider
        self._max_retries = settings.AI_PROVIDER_MAX_RETRIES
        self._backoff_seconds = settings.AI_PROVIDER_RETRY_BACKOFF_SECONDS

    @property
    def model_name(self) -> str:
        return self._provider.model_name

    def _new_retryer(self) -> AsyncRetrying:
        # A fresh AsyncRetrying per call — tenacity keeps per-attempt state on
        # the instance itself, so sharing one across concurrent calls (two
        # coroutines awaiting the same AIService instance at once) would let
        # them clobber each other's retry counters.
        return AsyncRetrying(
            retry=retry_if_exception_type(TransientProviderError),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=self._backoff_seconds),
            reraise=True,
        )

    async def extract_structured(
        self, *, text: str, schema: dict, system: str, max_tokens: int = 8000
    ) -> dict:
        async for attempt in self._new_retryer():
            with attempt:
                return await self._provider.extract_structured(
                    text=text, schema=schema, system=system, max_tokens=max_tokens
                )
        raise AssertionError("unreachable")  # tenacity always returns or raises above

    async def complete_chat(
        self, *, system: str, messages: list[dict[str, str]], max_tokens: int = 1500
    ) -> str:
        async for attempt in self._new_retryer():
            with attempt:
                return await self._provider.complete_chat(
                    system=system, messages=messages, max_tokens=max_tokens
                )
        raise AssertionError("unreachable")


__all__ = ["AIService"]
