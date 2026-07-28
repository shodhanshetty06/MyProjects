"""Unit tests for `AICareerCoachService` — deterministic scoring/grounding
logic, with `AIService` and every repository faked (duck-typed, no DB or
network required).

Coverage is deliberately weighted toward the logic that isn't just "call
the model and pass the result through": skill-gap grounding (and filtering
hallucinated skills out of a grounded response), job-readiness score
weighting, salary-prediction grounded vs. AI-estimate branching, and the
chat assistant's conversation/message persistence.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.core.exceptions import ConflictError, NotFoundError
from app.modules.ai_career_coach.models import CoachMessageRole
from app.modules.ai_career_coach.schemas import ResumeReviewResult
from app.modules.ai_career_coach.service import AICareerCoachService
from app.modules.resume_parser.models import ResumeVersionStatus


def _make_parsed_data(**overrides) -> SimpleNamespace:
    defaults = dict(
        summary="Experienced backend engineer.",
        skills=["Python", "SQL"],
        experience=[
            {
                "title": "Engineer",
                "company": "Acme",
                "start_date": "2020",
                "end_date": "2023",
                "description": "Built things",
                "highlights": ["Shipped X"],
            }
        ],
        education=[{"degree": "BSc", "field_of_study": "CS", "institution": "State University"}],
        projects=[],
        certifications=[],
        achievements=[],
        languages=[],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_resume_version(*, status=ResumeVersionStatus.PARSED, parsed_data=None) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        status=status,
        parsed_data=parsed_data if parsed_data is not None else _make_parsed_data(),
    )


class _FakeInsightRepository:
    def __init__(self, latest=None) -> None:
        self._latest = latest
        self.created: list[SimpleNamespace] = []

    async def get_latest(self, *, user_id, insight_type, resume_version_id=None):
        return self._latest

    async def create(self, **kwargs):
        record = SimpleNamespace(id=uuid.uuid4(), **kwargs)
        self.created.append(record)
        return record

    async def list_by_user(self, **kwargs):
        return []

    async def count_by_user(self, **kwargs):
        return 0

    async def get_owned(self, **kwargs):
        return None


class _FakeJobMarketRepository:
    def __init__(self, *, required_skills=None, salary_stats=None) -> None:
        self._required_skills = required_skills or []
        self._salary_stats = salary_stats or {
            "sample_size": 0,
            "avg_min": None,
            "avg_max": None,
            "currency": None,
        }

    async def required_skills_for_title(self, *, title_query, limit):
        return self._required_skills

    async def salary_stats_for_role(self, *, title_query, country, sample_limit):
        return self._salary_stats


class _FakeResumeContextRepository:
    def __init__(self, *, version=None, current_version=None) -> None:
        self._version = version
        self._current_version = current_version

    async def get_owned_version(self, *, user_id, resume_version_id):
        return self._version

    async def get_current_version_for_user(self, *, user_id):
        return self._current_version


class _FakeConversationRepository:
    def __init__(self, *, existing=None) -> None:
        self._existing = existing
        self.created: list[SimpleNamespace] = []
        self.updated: list[SimpleNamespace] = []

    async def get_owned(self, *, user_id, conversation_id, with_messages=False):
        return self._existing

    async def create(self, **kwargs):
        record = SimpleNamespace(id=uuid.uuid4(), **kwargs)
        self.created.append(record)
        return record

    async def update(self, instance, **kwargs):
        for key, value in kwargs.items():
            setattr(instance, key, value)
        self.updated.append(instance)
        return instance

    async def list_by_user(self, **kwargs):
        return []

    async def count_by_user(self, **kwargs):
        return 0


class _FakeMessageRepository:
    def __init__(self) -> None:
        self.created: list[SimpleNamespace] = []

    async def create(self, **kwargs):
        record = SimpleNamespace(id=uuid.uuid4(), created_at=datetime.now(timezone.utc), **kwargs)
        self.created.append(record)
        return record

    async def list_recent_for_conversation(self, *, conversation_id, limit):
        matching = [m for m in self.created if m.conversation_id == conversation_id]
        return matching[-limit:]


class _FakeAIService:
    def __init__(self, *, extract_responses=None, chat_responses=None) -> None:
        self._extract_responses = list(extract_responses or [])
        self._chat_responses = list(chat_responses or [])
        self.extract_calls: list[dict] = []
        self.chat_calls: list[dict] = []
        self.model_name = "fake-ai-model"

    async def extract_structured(self, *, text, schema, system, max_tokens=8000):
        self.extract_calls.append({"text": text, "system": system})
        return self._extract_responses.pop(0)

    async def complete_chat(self, *, system, messages, max_tokens=1500):
        self.chat_calls.append({"system": system, "messages": messages})
        return self._chat_responses.pop(0)


def _make_service(**overrides) -> AICareerCoachService:
    defaults = dict(
        insight_repository=_FakeInsightRepository(),
        conversation_repository=_FakeConversationRepository(),
        message_repository=_FakeMessageRepository(),
        resume_context_repository=_FakeResumeContextRepository(),
        job_market_repository=_FakeJobMarketRepository(),
        ai_service=_FakeAIService(),
    )
    defaults.update(overrides)
    return AICareerCoachService(**defaults)


# --- resume-gate ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_review_resume_raises_not_found_when_version_missing() -> None:
    service = _make_service(resume_context_repository=_FakeResumeContextRepository(version=None))

    with pytest.raises(NotFoundError):
        await service.review_resume(user_id=uuid.uuid4(), resume_version_id=uuid.uuid4())


@pytest.mark.asyncio
async def test_review_resume_rejects_unparsed_version() -> None:
    version = _make_resume_version(status=ResumeVersionStatus.UPLOADED, parsed_data=None)
    service = _make_service(resume_context_repository=_FakeResumeContextRepository(version=version))

    with pytest.raises(ConflictError):
        await service.review_resume(user_id=uuid.uuid4(), resume_version_id=version.id)


# --- skill gap analysis ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skill_gap_grounded_filters_ungrounded_ai_skills() -> None:
    version = _make_resume_version(parsed_data=_make_parsed_data(skills=["Python"]))
    ai = _FakeAIService(
        extract_responses=[
            {
                "missing_skills": [
                    {"skill": "Docker", "importance": "high", "reason": "Needed for deployment"},
                    {"skill": "Kubernetes", "importance": "medium", "reason": "not grounded"},
                ]
            }
        ]
    )
    service = _make_service(
        job_market_repository=_FakeJobMarketRepository(
            required_skills=[("Python", 10), ("Docker", 5)]
        ),
        ai_service=ai,
        resume_context_repository=_FakeResumeContextRepository(version=version),
    )

    result = await service.analyze_skill_gap(
        user_id=uuid.uuid4(), resume_version_id=version.id, target_role="Backend Engineer"
    )

    assert result.matched_skills == ["Python"]
    assert [item.skill for item in result.missing_skills] == ["Docker"]
    assert result.overall_readiness_percent == 50.0


@pytest.mark.asyncio
async def test_skill_gap_ungrounded_falls_back_to_ai_count_heuristic() -> None:
    version = _make_resume_version(parsed_data=_make_parsed_data(skills=["Python", "SQL"]))
    ai = _FakeAIService(
        extract_responses=[
            {
                "missing_skills": [
                    {"skill": "System Design", "importance": "high", "reason": "..."},
                    {"skill": "Leadership", "importance": "medium", "reason": "..."},
                ]
            }
        ]
    )
    service = _make_service(
        job_market_repository=_FakeJobMarketRepository(required_skills=[]),
        ai_service=ai,
        resume_context_repository=_FakeResumeContextRepository(version=version),
    )

    result = await service.analyze_skill_gap(
        user_id=uuid.uuid4(), resume_version_id=version.id, target_role="Staff Engineer"
    )

    assert result.matched_skills == ["Python", "SQL"]
    assert result.overall_readiness_percent == 80.0  # 100 - 2 missing * 10


# --- job readiness score ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_job_readiness_score_uses_configured_weights() -> None:
    version = _make_resume_version(
        parsed_data=_make_parsed_data(
            skills=["Python", "SQL", "Docker"],
            experience=[{"title": "Engineer"}],
            education=[{"degree": "BSc"}],
            projects=[{"name": "Proj"}],
            certifications=[{"name": "Cert"}],
        )
    )
    cached_review = SimpleNamespace(
        result=ResumeReviewResult(
            overall_score=80.0,
            summary="Good",
            strengths=["A"],
            weaknesses=["B"],
            section_feedback=[],
            improvement_suggestions=[],
        ).model_dump(mode="json")
    )
    ai = _FakeAIService(
        extract_responses=[{"narrative": "Solid readiness.", "recommendations": ["Practice interviews"]}]
    )
    service = _make_service(
        insight_repository=_FakeInsightRepository(latest=cached_review),
        ai_service=ai,
        resume_context_repository=_FakeResumeContextRepository(version=version),
    )

    result = await service.compute_job_readiness(
        user_id=uuid.uuid4(), resume_version_id=version.id, target_role=None
    )

    # skill_match_score = min(100, 3 skills * 8) = 24; completeness = 6/6 sections = 100
    expected_overall = round(80.0 * 0.4 + 24.0 * 0.35 + 100.0 * 0.25, 1)
    assert result.resume_quality_score == 80.0
    assert result.skill_match_score == 24.0
    assert result.completeness_score == 100.0
    assert result.overall_score == expected_overall
    assert result.recommendations == ["Practice interviews"]


# --- salary prediction -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_salary_grounded_uses_market_average() -> None:
    job_market = _FakeJobMarketRepository(
        salary_stats={
            "sample_size": 12,
            "avg_min": 90000.4321,
            "avg_max": 120000.789,
            "currency": "USD",
        }
    )
    ai = _FakeAIService(chat_responses=["This range reflects strong market demand."])
    service = _make_service(job_market_repository=job_market, ai_service=ai)

    result = await service.predict_salary(
        user_id=uuid.uuid4(), target_role="Backend Engineer", country="United States"
    )

    assert result.based_on_market_data is True
    assert result.predicted_min == 90000.43
    assert result.predicted_max == 120000.79
    assert result.currency == "USD"
    assert result.sample_size == 12
    assert not ai.extract_calls  # narrative-only call, no structured extraction needed


@pytest.mark.asyncio
async def test_predict_salary_ungrounded_uses_ai_estimate() -> None:
    job_market = _FakeJobMarketRepository(
        salary_stats={"sample_size": 1, "avg_min": None, "avg_max": None, "currency": None}
    )
    ai = _FakeAIService(
        extract_responses=[
            {
                "predicted_min": 50000,
                "predicted_max": 70000,
                "currency": "USD",
                "narrative": "Estimated from general knowledge.",
            }
        ]
    )
    service = _make_service(job_market_repository=job_market, ai_service=ai)

    result = await service.predict_salary(user_id=uuid.uuid4(), target_role="Niche Role")

    assert result.based_on_market_data is False
    assert result.predicted_min == 50000
    assert result.predicted_max == 70000
    assert not ai.chat_calls  # estimate-only call, no narrative-only chat needed


# --- chat assistant --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_message_creates_conversation_and_persists_both_turns() -> None:
    ai = _FakeAIService(chat_responses=["Here's some career advice."])
    conversation_repository = _FakeConversationRepository()
    message_repository = _FakeMessageRepository()
    service = _make_service(
        ai_service=ai,
        conversation_repository=conversation_repository,
        message_repository=message_repository,
    )

    conversation, assistant_message = await service.send_message(
        user_id=uuid.uuid4(),
        conversation_id=None,
        message="How do I switch careers into data science?",
    )

    assert conversation.title.startswith("How do I switch careers")
    assert len(message_repository.created) == 2
    assert message_repository.created[0].role == CoachMessageRole.USER
    assert message_repository.created[1].role == CoachMessageRole.ASSISTANT
    assert assistant_message.content == "Here's some career advice."
    assert conversation.last_message_at is not None
    assert ai.chat_calls[0]["messages"] == [
        {"role": "user", "content": "How do I switch careers into data science?"}
    ]


@pytest.mark.asyncio
async def test_send_message_raises_not_found_for_missing_conversation() -> None:
    service = _make_service(conversation_repository=_FakeConversationRepository(existing=None))

    with pytest.raises(NotFoundError):
        await service.send_message(
            user_id=uuid.uuid4(), conversation_id=uuid.uuid4(), message="hi"
        )
