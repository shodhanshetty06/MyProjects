"""Unit tests for `search/nl_query.py`'s parsing contract, using a fake
`AIProvider` (same convention as `tests/services/ai/test_ai_service.py`) so
no real LLM call is needed.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.modules.search.nl_query import ParsedSearchQuery, parse_natural_language_query
from app.services.ai.service import AIService


class _FakeProvider:
    def __init__(self, *, extraction: dict | None = None, raises: Exception | None = None) -> None:
        self._extraction = extraction or {}
        self._raises = raises
        self.model_name = "fake-model"

    async def extract_structured(self, *, text, schema, system, max_tokens=8000):
        if self._raises:
            raise self._raises
        return self._extraction

    async def complete_chat(self, *, system, messages, max_tokens=1500):
        raise NotImplementedError("not used by nl_query")


def _service(provider: _FakeProvider) -> AIService:
    return AIService(provider, Settings(AI_PROVIDER_MAX_RETRIES=1, AI_PROVIDER_RETRY_BACKOFF_SECONDS=0.001))


@pytest.mark.asyncio
async def test_empty_query_returns_empty_filters() -> None:
    result = await parse_natural_language_query("", ai_service=_service(_FakeProvider()))
    assert result == ParsedSearchQuery(cleaned_query="")


@pytest.mark.asyncio
async def test_parses_work_mode_skills_and_salary() -> None:
    provider = _FakeProvider(
        extraction={
            "work_mode": "remote",
            "skills": ["Python"],
            "salary_min": 2000000,
            "salary_max": None,
            "salary_currency": "inr",
            "salary_period": "yearly",
            "experience_level": None,
            "employment_type": None,
            "country": None,
            "city": None,
            "cleaned_query": "Python developer jobs",
        }
    )
    result = await parse_natural_language_query(
        "Find remote Python jobs paying above 20 LPA", ai_service=_service(provider)
    )
    assert result.work_mode == "remote"
    assert result.skills == ["Python"]
    assert result.salary_min == 2000000
    assert result.salary_currency == "INR"
    assert result.salary_period == "yearly"
    assert result.cleaned_query == "Python developer jobs"


@pytest.mark.asyncio
async def test_invalid_enum_values_are_dropped_not_fabricated() -> None:
    """A slightly-off LLM output (e.g. 'Remote' vs 'remote', or a
    hallucinated value not in the enum) should degrade to `None`, never
    raise or silently coerce to some other real value.
    """
    provider = _FakeProvider(
        extraction={
            "work_mode": "space station",  # not a real WorkMode
            "cleaned_query": "engineer jobs",
        }
    )
    result = await parse_natural_language_query("engineer jobs on mars", ai_service=_service(provider))
    assert result.work_mode is None
    assert result.cleaned_query == "engineer jobs"


@pytest.mark.asyncio
async def test_provider_failure_falls_back_to_original_query() -> None:
    provider = _FakeProvider(raises=RuntimeError("provider offline"))
    raw = "senior backend roles"
    result = await parse_natural_language_query(raw, ai_service=_service(provider))
    assert result.cleaned_query == raw
    assert result.work_mode is None
    assert result.skills == []


@pytest.mark.asyncio
async def test_blank_cleaned_query_falls_back_to_raw_query() -> None:
    provider = _FakeProvider(extraction={"cleaned_query": ""})
    raw = "python jobs"
    result = await parse_natural_language_query(raw, ai_service=_service(provider))
    assert result.cleaned_query == raw
