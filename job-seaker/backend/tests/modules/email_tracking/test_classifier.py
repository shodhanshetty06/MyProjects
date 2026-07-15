"""Unit tests for Claude email classification — the Anthropic client is
monkeypatched exactly like `tests/modules/resume_parser/test_ai_service.py`,
so no network access or API key is required.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.modules.email_tracking.classifier import classify_email
from app.services.ai.service import AIService


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


def _text_response(payload: dict):
    return SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
    )


@pytest.mark.asyncio
async def test_classify_email_parses_relevant_interview_email() -> None:
    payload = {
        "is_relevant": True,
        "classification": "interview",
        "confidence": 0.91,
        "extracted_company_name": "Acme Corp",
        "extracted_job_title": "Backend Engineer",
        "reasoning": "The email schedules an interview.",
    }
    ai_service = AIService()
    ai_service._anthropic_client = _FakeAnthropicClient(_text_response(payload))

    result = await classify_email(
        ai_service,
        sender="recruiting@acme.com",
        subject="Interview invite",
        snippet="We'd love to set up a call.",
    )

    assert result.is_relevant is True
    assert result.classification == "interview"
    assert result.confidence == 0.91
    assert result.extracted_company_name == "Acme Corp"

    call = ai_service._anthropic_client.messages.calls[0]
    assert "recruiting@acme.com" in call["messages"][0]["content"]
    assert call["output_config"]["format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_classify_email_maps_irrelevant_email_to_none() -> None:
    payload = {
        "is_relevant": False,
        "classification": "none",
        "confidence": 0.98,
        "extracted_company_name": None,
        "extracted_job_title": None,
        "reasoning": "This is a newsletter, not a status update.",
    }
    ai_service = AIService()
    ai_service._anthropic_client = _FakeAnthropicClient(_text_response(payload))

    result = await classify_email(
        ai_service,
        sender="newsletter@jobsite.com",
        subject="Weekly job digest",
        snippet="Check out these new openings.",
    )

    assert result.is_relevant is False
    assert result.classification == "none"
