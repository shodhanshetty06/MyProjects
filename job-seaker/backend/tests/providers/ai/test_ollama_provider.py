"""Unit tests for `OllamaProvider` — the `ollama.AsyncClient` is replaced
with a fake, so no running Ollama server is required. Covers: successful
structured extraction, connection-refused -> `TransientProviderError`
("Ollama offline"), and model-not-installed -> fallback-model retry.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import ollama
import pytest

from app.core.config import Settings
from app.core.exceptions import ProviderUnavailableError, TransientProviderError
from app.providers.ai.ollama_provider import OllamaProvider


def _settings(**overrides) -> Settings:
    defaults = dict(
        OLLAMA_MODEL="qwen3:8b", OLLAMA_FALLBACK_MODELS="llama3.1:8b,gemma3:12b"
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _chat_response(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))


class _FakeOllamaClient:
    def __init__(self, *, responses=None, exceptions=None) -> None:
        self._responses = list(responses or [])
        self._exceptions = list(exceptions or [])
        self.calls: list[dict] = []

    async def chat(self, *, model, messages, format=None, options=None):
        self.calls.append({"model": model, "messages": messages, "format": format})
        if self._exceptions:
            raise self._exceptions.pop(0)
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_extract_structured_returns_parsed_json() -> None:
    payload = {"summary": "Experienced engineer"}
    provider = OllamaProvider(_settings())
    provider._client = _FakeOllamaClient(responses=[_chat_response(payload)])

    result = await provider.extract_structured(text="resume", schema={"type": "object"}, system="sys")

    assert result == payload
    call = provider._client.calls[0]
    assert call["model"] == "qwen3:8b"
    assert call["format"] == {"type": "object"}


@pytest.mark.asyncio
async def test_connection_error_raises_transient_provider_error() -> None:
    provider = OllamaProvider(_settings())
    provider._client = _FakeOllamaClient(exceptions=[httpx.ConnectError("connection refused")])

    with pytest.raises(TransientProviderError):
        await provider.extract_structured(text="resume", schema={}, system="sys")


@pytest.mark.asyncio
async def test_model_not_installed_falls_back_to_next_model() -> None:
    payload = {"ok": True}
    provider = OllamaProvider(_settings(OLLAMA_FALLBACK_MODELS="llama3.1:8b"))
    provider._client = _FakeOllamaClient(
        exceptions=[ollama.ResponseError("model not found", 404)],
        responses=[_chat_response(payload)],
    )

    result = await provider.extract_structured(text="resume", schema={}, system="sys")

    assert result == payload
    assert [call["model"] for call in provider._client.calls] == ["qwen3:8b", "llama3.1:8b"]


@pytest.mark.asyncio
async def test_all_fallback_models_missing_raises_provider_unavailable() -> None:
    provider = OllamaProvider(_settings(OLLAMA_FALLBACK_MODELS="llama3.1:8b"))
    provider._client = _FakeOllamaClient(
        exceptions=[
            ollama.ResponseError("model not found", 404),
            ollama.ResponseError("model not found", 404),
        ]
    )

    with pytest.raises(ProviderUnavailableError):
        await provider.extract_structured(text="resume", schema={}, system="sys")


@pytest.mark.asyncio
async def test_complete_chat_returns_text() -> None:
    provider = OllamaProvider(_settings())
    provider._client = _FakeOllamaClient(
        responses=[SimpleNamespace(message=SimpleNamespace(content="Here's some advice."))]
    )

    result = await provider.complete_chat(system="sys", messages=[{"role": "user", "content": "hi"}])

    assert result == "Here's some advice."
