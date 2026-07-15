"""Unit tests for `AIService` — Claude structured extraction and Voyage
embeddings, with both provider clients monkeypatched. No network access or
API keys required.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.core.exceptions import ServiceUnavailableError
from app.services.ai.service import AIService


class _FakeVoyageResult:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings


class _FakeVoyageClient:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self._embeddings = embeddings
        self.calls: list[dict] = []

    def embed(self, texts, **kwargs):
        self.calls.append({"texts": texts, **kwargs})
        return _FakeVoyageResult(self._embeddings)


class _FakeAnthropicMessages:
    def __init__(self, response) -> None:
        self._response = response
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response) -> None:
        self.messages = _FakeAnthropicMessages(response)


def _text_response(payload: dict, stop_reason: str = "end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
    )


@pytest.mark.asyncio
async def test_generate_embedding_returns_first_vector() -> None:
    service = AIService()
    service._voyage_client = _FakeVoyageClient([[0.1, 0.2, 0.3]])

    result = await service.generate_embedding(text="Jane Doe, Software Engineer")

    assert result == [0.1, 0.2, 0.3]
    call = service._voyage_client.calls[0]
    assert call["texts"] == ["Jane Doe, Software Engineer"]
    assert call["input_type"] == "document"


@pytest.mark.asyncio
async def test_generate_embedding_wraps_provider_errors() -> None:
    class _RaisingVoyageClient:
        def embed(self, *args, **kwargs):
            raise RuntimeError("boom")

    service = AIService()
    service._voyage_client = _RaisingVoyageClient()

    with pytest.raises(ServiceUnavailableError):
        await service.generate_embedding(text="text")


@pytest.mark.asyncio
async def test_extract_structured_parses_json_text_block() -> None:
    payload = {"summary": "Experienced engineer", "skills": ["Python", "SQL"]}
    service = AIService()
    service._anthropic_client = _FakeAnthropicClient(_text_response(payload))

    result = await service.extract_structured(
        text="resume text", schema={"type": "object"}, system="system prompt"
    )

    assert result == payload
    call = service._anthropic_client.messages.calls[0]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["messages"] == [{"role": "user", "content": "resume text"}]


@pytest.mark.asyncio
async def test_extract_structured_raises_on_refusal() -> None:
    service = AIService()
    service._anthropic_client = _FakeAnthropicClient(_text_response({}, stop_reason="refusal"))

    with pytest.raises(ServiceUnavailableError):
        await service.extract_structured(text="t", schema={}, system="s")


@pytest.mark.asyncio
async def test_extract_structured_wraps_provider_errors() -> None:
    class _RaisingMessages:
        async def create(self, **kwargs):
            raise RuntimeError("boom")

    service = AIService()
    service._anthropic_client = SimpleNamespace(messages=_RaisingMessages())

    with pytest.raises(ServiceUnavailableError):
        await service.extract_structured(text="t", schema={}, system="s")
