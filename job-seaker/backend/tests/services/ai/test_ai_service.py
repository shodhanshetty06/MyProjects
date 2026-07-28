"""Unit tests for `AIService`'s retry behavior — provider-agnostic, so a
fake `AIProvider` (the `app/interfaces/ai_provider.py` contract) stands in
for any real backend. No network access or API key required.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.exceptions import ProviderUnavailableError, TransientProviderError
from app.services.ai.service import AIService


def _settings(**overrides) -> Settings:
    defaults = dict(AI_PROVIDER_MAX_RETRIES=3, AI_PROVIDER_RETRY_BACKOFF_SECONDS=0.001)
    defaults.update(overrides)
    return Settings(**defaults)


class _FlakyProvider:
    """Fails with `TransientProviderError` `fail_times` times, then succeeds."""

    def __init__(self, *, fail_times: int, result: dict | str = "ok") -> None:
        self._fail_times = fail_times
        self._result = result
        self.model_name = "fake-model"
        self.calls = 0

    async def extract_structured(self, *, text, schema, system, max_tokens=8000):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise TransientProviderError("offline")
        return self._result

    async def complete_chat(self, *, system, messages, max_tokens=1500):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise TransientProviderError("offline")
        return self._result


class _AlwaysUnavailableProvider:
    def __init__(self) -> None:
        self.model_name = "fake-model"
        self.calls = 0

    async def extract_structured(self, *, text, schema, system, max_tokens=8000):
        self.calls += 1
        raise ProviderUnavailableError("refused")

    async def complete_chat(self, *, system, messages, max_tokens=1500):
        self.calls += 1
        raise ProviderUnavailableError("refused")


@pytest.mark.asyncio
async def test_extract_structured_retries_transient_errors_then_succeeds() -> None:
    provider = _FlakyProvider(fail_times=2, result={"ok": True})
    service = AIService(provider, _settings())

    result = await service.extract_structured(text="t", schema={}, system="s")

    assert result == {"ok": True}
    assert provider.calls == 3


@pytest.mark.asyncio
async def test_extract_structured_raises_after_exhausting_retries() -> None:
    provider = _FlakyProvider(fail_times=10)
    service = AIService(provider, _settings(AI_PROVIDER_MAX_RETRIES=2))

    with pytest.raises(TransientProviderError):
        await service.extract_structured(text="t", schema={}, system="s")

    assert provider.calls == 2


@pytest.mark.asyncio
async def test_provider_unavailable_error_is_not_retried() -> None:
    provider = _AlwaysUnavailableProvider()
    service = AIService(provider, _settings(AI_PROVIDER_MAX_RETRIES=5))

    with pytest.raises(ProviderUnavailableError):
        await service.extract_structured(text="t", schema={}, system="s")

    # Non-retryable — only the first attempt should have run.
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_complete_chat_retries_and_succeeds() -> None:
    provider = _FlakyProvider(fail_times=1, result="hello")
    service = AIService(provider, _settings())

    result = await service.complete_chat(system="s", messages=[{"role": "user", "content": "hi"}])

    assert result == "hello"
    assert provider.calls == 2


def test_model_name_proxies_to_provider() -> None:
    provider = _FlakyProvider(fail_times=0)
    service = AIService(provider, _settings())

    assert service.model_name == "fake-model"
