"""Integration tests for the `/matching` API — preferences, on-demand match
computation, listing, and recommendation dismissal.

Needs a reachable Postgres (with resume_parser + job_ingestion + matching
tables migrated):

    docker compose up -d postgres redis
    docker compose exec api alembic upgrade head
    docker compose exec api pytest tests/modules/matching/test_router.py

Celery `.delay()` calls and the Redis-backed realtime publish are faked at
the function level (same approach as `resume_parser/test_router.py` faking
`parse_resume_task.delay`), so no running broker/worker is required. Each
test seeds its own randomized Resume/Job/embeddings directly through
`AsyncSessionLocal` so tests don't need to share or reset database state.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient

from app.core.config import get_settings
from app.core.security import create_access_token
from app.db.session import AsyncSessionLocal
from app.modules.job_ingestion.models import (
    ATSPlatform,
    Company,
    EmploymentType,
    Job,
    JobEmbedding,
    JobStatus,
    Location,
    Skill,
    WorkMode,
)
from app.modules.matching import realtime
from app.modules.matching.models import Recommendation, RecommendationStatus, RecommendationType
from app.modules.matching.tasks import compute_resume_matches_task, refresh_recommendations_task
from app.modules.resume_parser.models import (
    Resume,
    ResumeContentType,
    ResumeEmbedding,
    ResumeParsedData,
    ResumeVersion,
    ResumeVersionStatus,
)

VECTOR_DIMENSIONS = get_settings().VECTOR_DIMENSIONS


@pytest.fixture(autouse=True)
def _fake_external_dependencies(monkeypatch: pytest.MonkeyPatch):
    """No-op Celery enqueue + Redis publish — this suite only needs Postgres."""
    monkeypatch.setattr(compute_resume_matches_task, "delay", lambda *a, **k: None)
    monkeypatch.setattr(refresh_recommendations_task, "delay", lambda *a, **k: None)

    async def _fake_publish(user_id, event) -> None:  # noqa: ANN001
        return None

    monkeypatch.setattr(realtime, "publish_match_event", _fake_publish)
    yield


def _user_header(user_id: uuid.UUID) -> dict[str, str]:
    token, _ = create_access_token(user_id)
    return {"Authorization": f"Bearer {token}"}


def _embedding(seed: float) -> list[float]:
    return [seed] * VECTOR_DIMENSIONS


async def _seed_job(*, embedding: list[float], skill_names: list[str] | None = None) -> Job:
    async with AsyncSessionLocal() as session:
        company = Company(name="Acme Corp", normalized_name=f"acme-{uuid.uuid4()}")
        location = Location(
            raw_text="Berlin, Germany",
            city="Berlin",
            country="Germany",
            normalized_key=f"berlin-de-{uuid.uuid4()}",
        )
        session.add_all([company, location])
        await session.flush()

        job = Job(
            company_id=company.id,
            location_id=location.id,
            source=ATSPlatform.GREENHOUSE,
            source_job_id=str(uuid.uuid4()),
            source_url="https://example.com/jobs/1",
            title="Senior Backend Engineer",
            description="Build and operate backend services.",
            employment_type=EmploymentType.FULL_TIME,
            work_mode=WorkMode.REMOTE,
            status=JobStatus.ACTIVE,
            content_hash=str(uuid.uuid4()),
        )
        session.add(job)
        await session.flush()

        for name in skill_names or ["Python", "SQL"]:
            normalized = name.strip().lower()
            skill = Skill(name=name, normalized_name=normalized)
            session.add(skill)
            await session.flush()
            job.skills.append(skill)

        session.add(JobEmbedding(job_id=job.id, embedding=embedding, model_used="test-model"))
        await session.commit()
        await session.refresh(job)
        return job


async def _seed_resume_version(
    *, user_id: uuid.UUID, embedding: list[float], skills: list[str] | None = None
) -> ResumeVersion:
    async with AsyncSessionLocal() as session:
        resume = Resume(user_id=user_id, title="My Resume")
        session.add(resume)
        await session.flush()

        version = ResumeVersion(
            resume_id=resume.id,
            version_number=1,
            is_current=True,
            status=ResumeVersionStatus.PARSED,
            content_type=ResumeContentType.PDF,
            original_filename="resume.pdf",
            s3_bucket="test-bucket",
            s3_key=f"resumes/{uuid.uuid4()}.pdf",
            file_size_bytes=1024,
        )
        session.add(version)
        await session.flush()

        session.add(
            ResumeParsedData(
                resume_version_id=version.id,
                summary="Backend engineer.",
                skills=skills or ["Python"],
                model_used="test-model",
            )
        )
        session.add(
            ResumeEmbedding(
                resume_version_id=version.id, embedding=embedding, model_used="test-model"
            )
        )
        await session.commit()
        await session.refresh(version)
        return version


async def test_upsert_and_get_preferences(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    headers = _user_header(user_id)

    upsert_response = await client.put(
        "/api/v1/matching/preferences",
        headers=headers,
        json={
            "preferred_countries": ["Germany"],
            "preferred_work_modes": ["remote"],
            "preferred_employment_types": ["full_time"],
            "years_of_experience": 5,
            "expected_salary_min": 100000,
            "expected_salary_max": 150000,
            "salary_currency": "USD",
        },
    )
    assert upsert_response.status_code == 200
    assert upsert_response.json()["data"]["preferred_countries"] == ["Germany"]

    get_response = await client.get("/api/v1/matching/preferences", headers=headers)
    assert get_response.status_code == 200
    assert get_response.json()["data"]["years_of_experience"] == 5


async def test_get_preferences_404_when_unset(client: AsyncClient) -> None:
    response = await client.get("/api/v1/matching/preferences", headers=_user_header(uuid.uuid4()))
    assert response.status_code == 404


async def test_compute_single_match_returns_scores_and_narrative(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    job = await _seed_job(embedding=_embedding(0.5))
    version = await _seed_resume_version(
        user_id=user_id, embedding=_embedding(0.5), skills=["Python"]
    )

    response = await client.post(
        f"/api/v1/matching/resumes/{version.id}/jobs/{job.id}/compute",
        headers=_user_header(user_id),
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["job_id"] == str(job.id)
    assert data["status"] == "computed"
    assert 0 <= data["match_score"] <= 100
    assert "Python" in data["matched_skills"]
    assert "SQL" in data["missing_skills"]
    assert data["job"]["title"] == "Senior Backend Engineer"


async def test_list_resume_matches_after_compute(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    job = await _seed_job(embedding=_embedding(0.3))
    version = await _seed_resume_version(user_id=user_id, embedding=_embedding(0.3))

    headers = _user_header(user_id)
    await client.post(
        f"/api/v1/matching/resumes/{version.id}/jobs/{job.id}/compute", headers=headers
    )

    list_response = await client.get(
        f"/api/v1/matching/resumes/{version.id}/matches", headers=headers
    )
    assert list_response.status_code == 200
    body = list_response.json()
    assert body["pagination"]["total_items"] == 1
    assert body["data"][0]["job_id"] == str(job.id)


async def test_compute_batch_enqueues_and_returns_202(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    version = await _seed_resume_version(user_id=user_id, embedding=_embedding(0.1))

    response = await client.post(
        f"/api/v1/matching/resumes/{version.id}/compute", headers=_user_header(user_id)
    )
    assert response.status_code == 202


async def test_compute_batch_404_for_unknown_resume_version(client: AsyncClient) -> None:
    response = await client.post(
        f"/api/v1/matching/resumes/{uuid.uuid4()}/compute", headers=_user_header(uuid.uuid4())
    )
    assert response.status_code == 404


async def test_refresh_recommendations_returns_202(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/matching/recommendations/refresh", headers=_user_header(uuid.uuid4())
    )
    assert response.status_code == 202


async def test_dismiss_recommendation(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    job = await _seed_job(embedding=_embedding(0.2))

    async with AsyncSessionLocal() as session:
        recommendation = Recommendation(
            user_id=user_id,
            recommendation_type=RecommendationType.JOB,
            job_id=job.id,
            score=80,
            reason="Strong overall match score.",
        )
        session.add(recommendation)
        await session.commit()
        await session.refresh(recommendation)

    headers = _user_header(user_id)
    list_response = await client.get("/api/v1/matching/recommendations/jobs", headers=headers)
    assert list_response.status_code == 200
    assert len(list_response.json()["data"]) == 1

    dismiss_response = await client.post(
        f"/api/v1/matching/recommendations/{recommendation.id}/dismiss", headers=headers
    )
    assert dismiss_response.status_code == 200

    async with AsyncSessionLocal() as session:
        refreshed = await session.get(Recommendation, recommendation.id)
        assert refreshed is not None
        assert refreshed.status == RecommendationStatus.DISMISSED

    list_response_after = await client.get("/api/v1/matching/recommendations/jobs", headers=headers)
    assert list_response_after.json()["data"] == []
