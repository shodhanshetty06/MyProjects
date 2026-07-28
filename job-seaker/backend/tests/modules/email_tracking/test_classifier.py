"""Unit tests for AI email classification — a fake `AIProvider` (the
`app/interfaces/ai_provider.py` contract) stands in for a real LLM backend,
so no network access or API key is required.
"""

from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.modules.email_tracking.classifier import classify_email
from app.services.ai.service import AIService


class _FakeAIProvider:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.calls: list[dict] = []
        self.model_name = "fake-ai-model"

    async def extract_structured(self, *, text: str, schema: dict, system: str, max_tokens: int = 8000) -> dict:
        self.calls.append({"text": text, "schema": schema, "system": system})
        return self._payload

    async def complete_chat(self, *, system: str, messages: list[dict[str, str]], max_tokens: int = 1500) -> str:
        raise NotImplementedError


def _ai_service(payload: dict) -> tuple[AIService, _FakeAIProvider]:
    provider = _FakeAIProvider(payload)
    return AIService(provider, get_settings()), provider


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
    ai_service, provider = _ai_service(payload)

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

    call = provider.calls[0]
    assert "recruiting@acme.com" in call["text"]


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
    ai_service, _ = _ai_service(payload)

    result = await classify_email(
        ai_service,
        sender="newsletter@jobsite.com",
        subject="Weekly job digest",
        snippet="Check out these new openings.",
    )

    assert result.is_relevant is False
    assert result.classification == "none"
