"""Unit tests for `AnthropicProvider` — the `AsyncAnthropic` client is
replaced with a fake, so no API key or network access is required. Mirrors
the fake-client shape the pre-refactor `AIService` tests used.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.exceptions import ProviderUnavailableError, TransientProviderError
from app.providers.ai.anthropic_provider import AnthropicProvider


class _FakeAnthropicMessages:
    def __init__(self, response=None, exception=None) -> None:
        self._response = response
        self._exception = exception
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exception is not None:
            raise self._exception
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response=None, exception=None) -> None:
        self.messages = _FakeAnthropicMessages(response, exception)


def _text_response(payload: dict, stop_reason: str = "end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
    )


@pytest.mark.asyncio
async def test_extract_structured_parses_json_text_block() -> None:
    payload = {"summary": "Experienced engineer", "skills": ["Python", "SQL"]}
    provider = AnthropicProvider(Settings())
    provider._client_instance = _FakeAnthropicClient(_text_response(payload))

    result = await provider.extract_structured(text="resume text", schema={"type": "object"}, system="sys")

    assert result == payload
    call = provider._client.messages.calls[0]
    assert call["output_config"]["format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_extract_structured_raises_on_refusal() -> None:
    provider = AnthropicProvider(Settings())
    provider._client_instance = _FakeAnthropicClient(_text_response({}, stop_reason="refusal"))

    with pytest.raises(ProviderUnavailableError):
        await provider.extract_structured(text="t", schema={}, system="s")


@pytest.mark.asyncio
async def test_extract_structured_wraps_provider_errors_as_transient() -> None:
    provider = AnthropicProvider(Settings())
    provider._client_instance = _FakeAnthropicClient(exception=RuntimeError("connection reset"))

    with pytest.raises(TransientProviderError):
        await provider.extract_structured(text="t", schema={}, system="s")


@pytest.mark.asyncio
async def test_complete_chat_returns_text() -> None:
    provider = AnthropicProvider(Settings())
    response = SimpleNamespace(
        stop_reason="end_turn", content=[SimpleNamespace(type="text", text="Here's some advice.")]
    )
    provider._client_instance = _FakeAnthropicClient(response)

    result = await provider.complete_chat(system="s", messages=[{"role": "user", "content": "hi"}])

    assert result == "Here's some advice."


def test_model_name_reflects_settings() -> None:
    provider = AnthropicProvider(Settings(ANTHROPIC_MODEL="claude-opus-4-8"))

    assert provider.model_name == "claude-opus-4-8"
