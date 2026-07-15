"""Integration tests for the `/career-coach` API surface.

Unlike `resume_parser`'s `test_router.py`, this suite fakes the whole
`AICareerCoachService` (via the `get_ai_career_coach_service` FastAPI
dependency) rather than a real DB-backed one — every use case here is a
thin request/response/validation contract around a service call that's
already covered at the logic level in `test_service.py`, so there's
nothing DB-specific to exercise. No Postgres, Redis, or Anthropic/Voyage
credentials are required to run this file.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from httpx import AsyncClient

from app.core.exceptions import NotFoundError
from app.core.security import create_access_token
from app.main import app
from app.modules.ai_career_coach.models import CoachMessageRole
from app.modules.ai_career_coach.router import get_ai_career_coach_service
from app.modules.ai_career_coach.schemas import (
    CareerRoadmapResult,
    CareerSuggestionResult,
    CountryDemandResult,
    InterviewPrepResult,
    JobReadinessScoreResult,
    LearningRecommendationResult,
    MarketTrendResult,
    ResumeReviewResult,
    SalaryPredictionResult,
    SkillGapResult,
    SuggestedRole,
)


class _FakeCoachService:
    async def review_resume(self, *, user_id, resume_version_id):
        return ResumeReviewResult(
            overall_score=75.0,
            summary="Solid resume.",
            strengths=["Clear structure"],
            weaknesses=["Needs metrics"],
            section_feedback=[],
            improvement_suggestions=[],
        )

    async def suggest_careers(self, *, user_id, resume_version_id):
        return CareerSuggestionResult(
            summary="Good fit for backend roles.",
            suggested_roles=[
                SuggestedRole(
                    title="Backend Engineer", fit_score=88.0, match_reason="Strong Python background"
                )
            ],
        )

    async def analyze_skill_gap(self, *, user_id, resume_version_id, target_role):
        return SkillGapResult(
            target_role=target_role, matched_skills=["Python"], missing_skills=[], overall_readiness_percent=100.0
        )

    async def recommend_learning(self, *, user_id, resume_version_id, target_role, skills=None):
        return LearningRecommendationResult(target_role=target_role, recommendations=[])

    async def generate_roadmap(self, *, user_id, resume_version_id, target_role, timeframe_months):
        return CareerRoadmapResult(target_role=target_role, timeframe_months=timeframe_months, milestones=[])

    async def prepare_interview(self, *, user_id, resume_version_id, target_role):
        return InterviewPrepResult(target_role=target_role, questions=[], general_tips=[])

    async def compute_job_readiness(self, *, user_id, resume_version_id, target_role=None):
        return JobReadinessScoreResult(
            overall_score=70.0,
            resume_quality_score=75.0,
            skill_match_score=60.0,
            completeness_score=80.0,
            narrative="Good progress.",
            recommendations=[],
        )

    async def predict_salary(self, *, user_id, target_role, country=None, years_experience=None, resume_version_id=None):
        return SalaryPredictionResult(
            target_role=target_role,
            country=country,
            predicted_min=90000,
            predicted_max=120000,
            currency="USD",
            based_on_market_data=True,
            sample_size=10,
            narrative="Competitive range.",
        )

    async def analyze_market_trends(self, *, user_id, title=None, window_days=None):
        return MarketTrendResult(
            window_days=window_days or 90,
            title=title,
            total_postings=100,
            previous_window_postings=80,
            growth_percent=25.0,
            top_skills=[],
            top_companies=[],
            narrative="Demand is growing.",
        )

    async def analyze_country_demand(self, *, user_id, title=None, skill=None, limit=None):
        return CountryDemandResult(title=title, skill=skill, countries=[], narrative="Demand concentrated in a few markets.")

    async def list_insights(self, *, user_id, insight_type, offset, limit):
        return [], 0

    async def get_insight(self, *, user_id, insight_id):
        raise NotFoundError("Insight not found.")

    async def send_message(self, *, user_id, conversation_id, message):
        now = datetime.now(timezone.utc)
        conversation = SimpleNamespace(id=uuid.uuid4(), title="New conversation", last_message_at=now, created_at=now)
        assistant_message = SimpleNamespace(
            id=uuid.uuid4(), role=CoachMessageRole.ASSISTANT, content="Here's some advice.", created_at=now
        )
        return conversation, assistant_message

    async def list_conversations(self, *, user_id, offset, limit):
        return [], 0

    async def get_conversation(self, *, user_id, conversation_id):
        raise NotFoundError("Conversation not found.")

    async def delete_conversation(self, *, user_id, conversation_id):
        return None


@pytest.fixture(autouse=True)
def _fake_service_override():
    app.dependency_overrides[get_ai_career_coach_service] = lambda: _FakeCoachService()
    yield
    app.dependency_overrides.pop(get_ai_career_coach_service, None)


def _headers() -> dict[str, str]:
    token, _ = create_access_token(uuid.uuid4())
    return {"Authorization": f"Bearer {token}"}


async def test_missing_user_header_returns_401(client: AsyncClient) -> None:
    response = await client.get("/api/v1/career-coach/market-trends")
    assert response.status_code == 401


async def test_review_resume_returns_generated_result(client: AsyncClient) -> None:
    response = await client.post(
        f"/api/v1/career-coach/resumes/{uuid.uuid4()}/review", headers=_headers()
    )
    assert response.status_code == 200
    assert response.json()["data"]["overall_score"] == 75.0


async def test_skill_gap_requires_target_role_in_body(client: AsyncClient) -> None:
    response = await client.post(
        f"/api/v1/career-coach/resumes/{uuid.uuid4()}/skill-gap", headers=_headers(), json={}
    )
    assert response.status_code == 422


async def test_skill_gap_success(client: AsyncClient) -> None:
    response = await client.post(
        f"/api/v1/career-coach/resumes/{uuid.uuid4()}/skill-gap",
        headers=_headers(),
        json={"target_role": "Backend Engineer"},
    )
    assert response.status_code == 200
    assert response.json()["data"]["target_role"] == "Backend Engineer"


async def test_salary_prediction_success(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/career-coach/salary-prediction",
        headers=_headers(),
        json={"target_role": "Backend Engineer", "country": "United States"},
    )
    assert response.status_code == 200
    body = response.json()["data"]
    assert body["based_on_market_data"] is True
    assert body["predicted_min"] == 90000


async def test_market_trends_success(client: AsyncClient) -> None:
    response = await client.get(
        "/api/v1/career-coach/market-trends", headers=_headers(), params={"window_days": 30}
    )
    assert response.status_code == 200
    assert response.json()["data"]["total_postings"] == 100


async def test_get_insight_not_found_maps_to_404(client: AsyncClient) -> None:
    response = await client.get(f"/api/v1/career-coach/insights/{uuid.uuid4()}", headers=_headers())
    assert response.status_code == 404
    assert response.json()["error"]["error_code"] == "not_found"


async def test_list_insights_returns_paginated_envelope(client: AsyncClient) -> None:
    response = await client.get("/api/v1/career-coach/insights", headers=_headers())
    assert response.status_code == 200
    body = response.json()
    assert body["data"] == []
    assert body["pagination"]["total_items"] == 0


async def test_send_chat_message_creates_reply(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/career-coach/chat",
        headers=_headers(),
        json={"message": "How do I negotiate salary?"},
    )
    assert response.status_code == 200
    body = response.json()["data"]
    assert body["message"]["role"] == "assistant"
    assert body["message"]["content"] == "Here's some advice."


async def test_chat_requires_nonempty_message(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/career-coach/chat", headers=_headers(), json={"message": ""}
    )
    assert response.status_code == 422


async def test_get_conversation_not_found_maps_to_404(client: AsyncClient) -> None:
    response = await client.get(
        f"/api/v1/career-coach/conversations/{uuid.uuid4()}", headers=_headers()
    )
    assert response.status_code == 404
    assert response.json()["error"]["error_code"] == "not_found"


async def test_delete_conversation_returns_success_message(client: AsyncClient) -> None:
    response = await client.delete(
        f"/api/v1/career-coach/conversations/{uuid.uuid4()}", headers=_headers()
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Conversation deleted."
