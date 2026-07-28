"""End-to-end "Ollama is offline" test: `OllamaProvider` raising
`TransientProviderError` on every attempt (connection refused) should
exhaust `AIService`'s retries and surface as `ServiceUnavailableError` —
the exception every route/task already knows how to handle (503 via the
global exception middleware), so offline mode degrades gracefully instead
of crashing a request/task with a raw connection error.
"""

from __future__ import annotations

import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import ServiceUnavailableError, TransientProviderError
from app.providers.ai.ollama_provider import OllamaProvider
from app.services.ai.service import AIService


class _AlwaysOfflineOllamaClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, *, model, messages, format=None, options=None):
        self.calls += 1
        raise httpx.ConnectError("connection refused")


@pytest.mark.asyncio
async def test_ollama_offline_exhausts_retries_and_raises_service_unavailable() -> None:
    settings = Settings(AI_PROVIDER_MAX_RETRIES=3, AI_PROVIDER_RETRY_BACKOFF_SECONDS=0.001)
    provider = OllamaProvider(settings)
    fake_client = _AlwaysOfflineOllamaClient()
    provider._client = fake_client

    service = AIService(provider, settings)

    with pytest.raises(ServiceUnavailableError):
        await service.extract_structured(text="resume", schema={}, system="sys")

    # TransientProviderError is retried up to AI_PROVIDER_MAX_RETRIES times.
    assert fake_client.calls == 3


@pytest.mark.asyncio
async def test_transient_provider_error_is_a_service_unavailable_error() -> None:
    # ServiceUnavailableError is the base every provider exception maps to,
    # so callers only need to handle one exception type for "AI is down."
    assert issubclass(TransientProviderError, ServiceUnavailableError)
