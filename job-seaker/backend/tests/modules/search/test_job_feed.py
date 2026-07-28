"""Integration tests for the Job Feed composition surface
(`/search/feed`, `/search/jobs/{id}`, `/search/jobs/{id}/why-score`,
`/search/home-summary`).

Needs a reachable Postgres (job_ingestion + search + matching + tracking +
resume_parser + workspace tables migrated) and Redis:

    docker compose exec api alembic upgrade head
    docker compose exec api pytest tests/modules/search/test_job_feed.py

Embeddings are faked via a dependency override on `get_search_service` (same
approach as `search/test_router.py`); Redis publish is faked at the function
level (same approach as `matching/test_router.py`). Each test seeds its own
randomized rows directly through `AsyncSessionLocal`.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient

from app.api.deps import CacheDep, DBSessionDep
from app.core.config import get_settings
from app.core.security import create_access_token
from app.db.session import AsyncSessionLocal
from app.main import app
from app.modules.job_ingestion.models import (
    ATSPlatform,
    Company,
    EmploymentType,
    Job,
    JobEmbedding,
    JobStatus,
    WorkMode,
)
from app.modules.matching.models import MatchStatus, ResumeJobMatch
from app.modules.search import realtime
from app.modules.search.repository import SearchRepository
from app.modules.search.router import get_search_service
from app.modules.search.service import SearchService
from app.modules.tracking.models import Application, ApplicationSource, ApplicationStatus
from app.services.embedding.service import EmbeddingService

VECTOR_DIMENSIONS = get_settings().VECTOR_DIMENSIONS


class _FakeEmbeddingProvider:
    def __init__(self, embedding: list[float]) -> None:
        self._embedding = embedding

    @property
    def dimensions(self) -> int:
        return len(self._embedding)

    @property
    def model_name(self) -> str:
        return "fake-embedding-model"

    async def embed(self, *, text: str, input_type: str = "document") -> list[float]:
        return self._embedding


@pytest.fixture(autouse=True)
def _override_search_service():
    settings = get_settings()

    def _get_search_service(session: DBSessionDep, cache: CacheDep) -> SearchService:
        provider = _FakeEmbeddingProvider([0.1] * VECTOR_DIMENSIONS)
        embedding_service = EmbeddingService(provider, settings)
        return SearchService(
            repository=SearchRepository(session), embedding_service=embedding_service, cache=cache
        )

    app.dependency_overrides[get_search_service] = _get_search_service
    yield
    app.dependency_overrides.pop(get_search_service, None)


@pytest.fixture(autouse=True)
def _fake_realtime(monkeypatch: pytest.MonkeyPatch):
    async def _fake_publish(user_id, event) -> None:  # noqa: ANN001
        return None

    monkeypatch.setattr(realtime, "publish_search_event", _fake_publish)
    yield


def _unique(prefix: str = "token") -> str:
    return f"{prefix}{uuid.uuid4().hex[:10]}"


def _user_header(user_id: uuid.UUID) -> dict[str, str]:
    token, _ = create_access_token(user_id)
    return {"Authorization": f"Bearer {token}"}


async def _seed_job(
    *,
    title: str | None = None,
    posted_at: datetime | None = None,
    skills: list[str] | None = None,
    company_name: str | None = None,
) -> Job:
    async with AsyncSessionLocal() as session:
        company = Company(name=company_name or _unique("Company "), normalized_name=_unique("company-"))
        session.add(company)
        await session.flush()

        job = Job(
            company_id=company.id,
            source=ATSPlatform.CUSTOM_HTML,
            source_job_id=_unique("job-"),
            source_url="https://example.com/job",
            title=title or _unique("Engineer "),
            description="Build things with Python and SQL.",
            employment_type=EmploymentType.FULL_TIME,
            work_mode=WorkMode.REMOTE,
            posted_at=posted_at or datetime.now(UTC),
            status=JobStatus.ACTIVE,
            content_hash=_unique("hash-"),
        )
        session.add(job)
        await session.flush()

        # Inserted via the raw `job_skill` association table (not
        # `job.skills.append(...)`) — this container's installed
        # SQLAlchemy/asyncpg combination raises `MissingGreenlet` on a
        # collection lazy-load triggered by appending to a many-to-many
        # relationship right after `flush()` (reproduces even outside
        # pytest/httpx, with plain `asyncio.run()` — a pre-existing
        # environment defect, not specific to this test file). Get-or-create
        # by `normalized_name` since `Skill.normalized_name` is globally
        # unique and this suite reuses literal names (e.g. "Python") both
        # within one test (two jobs sharing a skill) and across the
        # persistent dev DB shared by every prior test run.
        from sqlalchemy import insert, select as sa_select

        from app.modules.job_ingestion.models import Skill, job_skill

        for name in skills or []:
            normalized = name.strip().lower()
            existing = (
                await session.execute(
                    sa_select(Skill).where(Skill.normalized_name == normalized)
                )
            ).scalars().one_or_none()
            if existing is None:
                skill = Skill(name=name, normalized_name=normalized)
                session.add(skill)
                await session.flush()
                skill_id = skill.id
            else:
                skill_id = existing.id
            await session.execute(insert(job_skill).values(job_id=job.id, skill_id=skill_id))

        session.add(JobEmbedding(job_id=job.id, embedding=[0.1] * VECTOR_DIMENSIONS, model_used="t"))
        await session.commit()
        await session.refresh(job)
        return job


async def _seed_resume_version(*, user_id: uuid.UUID):
    from app.modules.resume_parser.models import (
        Resume,
        ResumeContentType,
        ResumeEmbedding,
        ResumeParsedData,
        ResumeVersion,
        ResumeVersionStatus,
    )

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
            ats_baseline_score=70.0,
        )
        session.add(version)
        await session.flush()

        session.add(
            ResumeParsedData(
                resume_version_id=version.id, summary="Backend engineer.", skills=["Python"], model_used="t"
            )
        )
        session.add(
            ResumeEmbedding(
                resume_version_id=version.id, embedding=[0.1] * VECTOR_DIMENSIONS, model_used="t"
            )
        )
        await session.commit()
        await session.refresh(version)
        return version


async def _seed_match(
    *, user_id: uuid.UUID, resume_version_id: uuid.UUID, job_id: uuid.UUID, match_score: float = 82.0
) -> ResumeJobMatch:
    async with AsyncSessionLocal() as session:
        match = ResumeJobMatch(
            resume_version_id=resume_version_id,
            job_id=job_id,
            user_id=user_id,
            status=MatchStatus.COMPUTED,
            match_score=match_score,
            semantic_score=0.8,
            skill_match_score=75.0,
            rule_boost_score=6.0,
            matched_skills=["Python"],
            missing_skills=["SQL"],
            strengths=["Strong Python match."],
            weaknesses=["Missing SQL."],
            score_breakdown={
                "semantic_score": 0.8,
                "skill_match_score": 75.0,
                "rule_boost_score": 6.0,
                "boosts": {"country": 6.0, "work_mode": None, "salary": 0.0, "experience": None},
            },
        )
        session.add(match)
        await session.commit()
        await session.refresh(match)
        return match


async def test_feed_recommended_uses_matches(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    job = await _seed_job(skills=["Python", "SQL"])
    version = await _seed_resume_version(user_id=user_id)
    await _seed_match(user_id=user_id, resume_version_id=version.id, job_id=job.id)

    response = await client.get(
        "/api/v1/search/feed", params={"feed_mode": "recommended"}, headers=_user_header(user_id)
    )
    assert response.status_code == 200
    body = response.json()
    ids = [item["id"] for item in body["data"]]
    assert str(job.id) in ids
    item = next(i for i in body["data"] if i["id"] == str(job.id))
    assert item["match"]["match_score"] == 82.0
    assert item["ai_signals"]["application_difficulty"] in {"easy", "moderate", "hard"}
    assert item["ai_signals"]["expected_ats_score"] is not None


async def test_feed_recommended_falls_back_to_newest_when_no_matches(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    job = await _seed_job(title=f"Fallback Job {_unique()}")

    response = await client.get(
        "/api/v1/search/feed", params={"feed_mode": "recommended"}, headers=_user_header(user_id)
    )
    assert response.status_code == 200
    body = response.json()
    ids = [item["id"] for item in body["data"]]
    assert str(job.id) in ids
    item = next(i for i in body["data"] if i["id"] == str(job.id))
    assert item["match"] is None


async def test_feed_all_mode(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    token = _unique("all")
    job = await _seed_job(title=f"All Mode Job {token}")

    response = await client.get(
        "/api/v1/search/feed",
        params={"feed_mode": "all", "q": token},
        headers=_user_header(user_id),
    )
    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["data"]]
    assert str(job.id) in ids


async def test_feed_newest_mode_orders_by_posted_at(client: AsyncClient) -> None:
    # `feed_mode=newest` delegates to `SearchService.search`, whose results
    # are cached by `filters` (see `SearchService.search`'s `cache_key`) —
    # a unique `company_name` filter (not just a unique title) keeps this
    # test's cache key distinct from any other run against the same Redis
    # instance. `company_name` (a plain scalar query field), not `skills`
    # (a `list[str]` query field) — this FastAPI/pydantic version does not
    # bind list-typed fields through a bare `Depends()`-bound Pydantic
    # model for *any* endpoint in this codebase (confirmed against the
    # pre-existing, unmodified `/search?work_mode=...` filter too — a
    # pre-existing defect, not introduced by this feed layer).
    user_id = uuid.uuid4()
    token = _unique("newest")
    older = await _seed_job(
        title=f"Older {token}",
        posted_at=datetime.now(UTC) - timedelta(days=3),
        company_name=f"Newest Co {token}",
    )
    newer = await _seed_job(
        title=f"Newer {token}",
        posted_at=datetime.now(UTC) - timedelta(hours=1),
        company_name=f"Newest Co {token}",
    )

    response = await client.get(
        "/api/v1/search/feed",
        params={"feed_mode": "newest", "page_size": 50, "company_name": f"Newest Co {token}"},
        headers=_user_header(user_id),
    )
    ids = [item["id"] for item in response.json()["data"]]
    assert ids.index(str(newer.id)) < ids.index(str(older.id))


async def test_feed_saved_mode(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    job = await _seed_job()
    headers = _user_header(user_id)

    await client.post(f"/api/v1/search/saved-jobs/{job.id}", headers=headers)
    response = await client.get(
        "/api/v1/search/feed", params={"feed_mode": "saved"}, headers=headers
    )
    assert response.status_code == 200
    body = response.json()
    assert body["pagination"]["total_items"] == 1
    assert body["data"][0]["is_saved"] is True


async def test_feed_applied_mode(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    job = await _seed_job()

    async with AsyncSessionLocal() as session:
        application = Application(
            user_id=user_id,
            job_id=job.id,
            company_name="Acme",
            job_title=job.title,
            status=ApplicationStatus.APPLIED,
            source=ApplicationSource.MANUAL,
        )
        session.add(application)
        await session.commit()

    response = await client.get(
        "/api/v1/search/feed", params={"feed_mode": "applied"}, headers=_user_header(user_id)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["pagination"]["total_items"] == 1
    assert body["data"][0]["application_status"] == "applied"


async def test_feed_trending_mode(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    token = _unique("trend")
    job = await _seed_job(title=f"Trending {token}")

    response = await client.get(
        "/api/v1/search/feed", params={"feed_mode": "trending", "page_size": 50}, headers=_user_header(user_id)
    )
    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["data"]]
    assert str(job.id) in ids


async def test_job_detail_anonymous_access(client: AsyncClient) -> None:
    job = await _seed_job()
    response = await client.get(f"/api/v1/search/jobs/{job.id}")
    assert response.status_code == 200
    body = response.json()["data"]
    assert body["is_saved"] is False
    assert body["application_status"] is None


async def test_job_detail_404_for_unknown_job(client: AsyncClient) -> None:
    response = await client.get(f"/api/v1/search/jobs/{uuid.uuid4()}")
    assert response.status_code == 404


async def test_why_score_returns_not_evaluated_education(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    job = await _seed_job(skills=["Python", "SQL"])
    version = await _seed_resume_version(user_id=user_id)
    await _seed_match(user_id=user_id, resume_version_id=version.id, job_id=job.id)

    response = await client.get(
        f"/api/v1/search/jobs/{job.id}/why-score", headers=_user_header(user_id)
    )
    assert response.status_code == 200
    body = response.json()["data"]
    assert body["education_match"] == "Not evaluated yet"
    assert "Python" in body["matched_skills"]
    assert "SQL" in body["missing_skills"]


async def test_why_score_no_match_yet(client: AsyncClient) -> None:
    job = await _seed_job()
    response = await client.get(
        f"/api/v1/search/jobs/{job.id}/why-score", headers=_user_header(uuid.uuid4())
    )
    assert response.status_code == 200
    body = response.json()["data"]
    assert body["matched_skills"] == []
    assert body["education_match"] == "Not evaluated yet"


async def test_home_summary(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    response = await client.get("/api/v1/search/home-summary", headers=_user_header(user_id))
    assert response.status_code == 200
    body = response.json()["data"]
    assert "greeting" in body
    assert isinstance(body["jobs_matching_profile_count"], int)
    assert isinstance(body["new_jobs_since_yesterday"], int)
