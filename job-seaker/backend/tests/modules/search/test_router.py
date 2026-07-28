"""Integration tests for the `/search` API — filters, keyword/semantic/
hybrid modes, sorting, autocomplete, suggestions, trending, and caching.

Needs a reachable Postgres (with job_ingestion + search tables/indexes
migrated) and Redis:

    docker compose up -d postgres redis
    docker compose exec api alembic upgrade head
    docker compose exec api pytest tests/modules/search/test_router.py

Embeddings are faked at the `EmbeddingProvider` level (a fake implementing
the `embed`/`dimensions`/`model_name` contract from
`app/interfaces/embedding_provider.py`) via a dependency override on
`get_search_service`, so no API key, network access, or local model
download is required. Each test seeds its own randomized Company/Job rows
directly through `AsyncSessionLocal` (job_ingestion exposes no "create job"
endpoint — jobs are normally created by the crawl pipeline) so tests don't
need to share or reset database state.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient

from app.api.deps import CacheDep, DBSessionDep
from app.core.config import get_settings
from app.db.session import AsyncSessionLocal
from app.main import app
from app.modules.job_ingestion.models import (
    ATSPlatform,
    Company,
    EmploymentType,
    Job,
    JobEmbedding,
    JobStatus,
    Location,
    SalaryPeriod,
    Skill,
    WorkMode,
)
from app.modules.search.repository import SearchRepository
from app.modules.search.router import get_search_service
from app.modules.search.service import SearchService
from app.services.embedding.service import EmbeddingService

VECTOR_DIMENSIONS = get_settings().VECTOR_DIMENSIONS


class _FakeEmbeddingProvider:
    def __init__(self, embedding: list[float]) -> None:
        self._embedding = embedding
        self.calls: list[dict] = []

    @property
    def dimensions(self) -> int:
        return len(self._embedding)

    @property
    def model_name(self) -> str:
        return "fake-embedding-model"

    async def embed(self, *, text: str, input_type: str = "document") -> list[float]:
        self.calls.append({"text": text, "input_type": input_type})
        return self._embedding


@pytest.fixture
def fake_embedding_provider() -> _FakeEmbeddingProvider:
    return _FakeEmbeddingProvider([0.1] * VECTOR_DIMENSIONS)


@pytest.fixture(autouse=True)
def _override_search_service(fake_embedding_provider: _FakeEmbeddingProvider):
    settings = get_settings()

    def _get_search_service(session: DBSessionDep, cache: CacheDep) -> SearchService:
        embedding_service = EmbeddingService(fake_embedding_provider, settings)
        return SearchService(
            repository=SearchRepository(session), embedding_service=embedding_service, cache=cache
        )

    app.dependency_overrides[get_search_service] = _get_search_service
    yield
    app.dependency_overrides.pop(get_search_service, None)


def _unique(prefix: str = "token") -> str:
    return f"{prefix}{uuid.uuid4().hex[:10]}"


async def _seed_job(
    *,
    title: str | None = None,
    description: str = "Build things with Python and SQL.",
    work_mode: WorkMode = WorkMode.REMOTE,
    employment_type: EmploymentType = EmploymentType.FULL_TIME,
    salary_min: float | None = None,
    salary_max: float | None = None,
    salary_currency: str | None = None,
    salary_period: SalaryPeriod = SalaryPeriod.UNKNOWN,
    posted_at: datetime | None = None,
    skills: list[str] | None = None,
    city: str | None = None,
    region: str | None = None,
    country: str | None = None,
    embedding: list[float] | None = None,
    company_name: str | None = None,
) -> Job:
    async with AsyncSessionLocal() as session:
        company = Company(
            name=company_name or _unique("Company "),
            normalized_name=_unique("company-"),
        )
        session.add(company)
        await session.flush()

        location = None
        if city or region or country:
            location = Location(
                raw_text=f"{city or ''} {region or ''} {country or ''}".strip(),
                city=city,
                region=region,
                country=country,
                normalized_key=_unique("location-"),
            )
            session.add(location)
            await session.flush()

        job = Job(
            company_id=company.id,
            location_id=location.id if location else None,
            source=ATSPlatform.CUSTOM_HTML,
            source_job_id=_unique("job-"),
            source_url="https://example.com/job",
            title=title or _unique("Engineer "),
            description=description,
            employment_type=employment_type,
            work_mode=work_mode,
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency=salary_currency,
            salary_period=salary_period,
            posted_at=posted_at or datetime.now(UTC),
            status=JobStatus.ACTIVE,
            content_hash=_unique("hash-"),
        )
        session.add(job)
        await session.flush()

        for name in skills or []:
            skill = Skill(name=name, normalized_name=name.strip().lower())
            session.add(skill)
            await session.flush()
            job.skills.append(skill)

        if embedding is not None:
            session.add(JobEmbedding(job_id=job.id, embedding=embedding, model_used="voyage-3.5"))

        await session.commit()
        await session.refresh(job)
        return job


@pytest.mark.asyncio
async def test_search_no_query_browses_with_pagination(client: AsyncClient) -> None:
    await _seed_job()
    response = await client.get("/api/v1/search", params={"page": 1, "page_size": 5})
    assert response.status_code == 200
    body = response.json()
    assert "pagination" in body
    assert body["pagination"]["page"] == 1


@pytest.mark.asyncio
async def test_keyword_search_finds_job_by_title_token(client: AsyncClient) -> None:
    token = _unique("kw")
    await _seed_job(title=f"Senior Platform Engineer {token}")

    response = await client.get("/api/v1/search", params={"q": token, "mode": "keyword"})
    assert response.status_code == 200
    body = response.json()
    assert body["pagination"]["total_items"] >= 1
    result = next(item for item in body["data"] if token in item["title"])
    assert result["keyword_score"] is not None
    assert result["match_score"] is None


@pytest.mark.asyncio
async def test_semantic_search_finds_job_by_embedding(
    client: AsyncClient, fake_embedding_provider: _FakeEmbeddingProvider
) -> None:
    token = _unique("sem")
    job = await _seed_job(title=f"Data Scientist {token}", embedding=[0.1] * VECTOR_DIMENSIONS)

    response = await client.get("/api/v1/search", params={"q": token, "mode": "semantic"})
    assert response.status_code == 200
    body = response.json()
    ids = [item["id"] for item in body["data"]]
    assert str(job.id) in ids
    result = next(item for item in body["data"] if item["id"] == str(job.id))
    assert result["match_score"] is not None
    assert fake_embedding_provider.calls[-1]["input_type"] == "query"


@pytest.mark.asyncio
async def test_match_score_sort_rejected_with_keyword_mode(client: AsyncClient) -> None:
    response = await client.get(
        "/api/v1/search",
        params={"q": "engineer", "mode": "keyword", "sort_by": "match_score"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_work_mode_filter(client: AsyncClient) -> None:
    token = _unique("wm")
    await _seed_job(title=f"Remote Job {token}", work_mode=WorkMode.REMOTE)
    await _seed_job(title=f"Onsite Job {token}", work_mode=WorkMode.ONSITE)

    response = await client.get("/api/v1/search", params={"work_mode": "remote", "page_size": 50})
    assert response.status_code == 200
    titles = [item["title"] for item in response.json()["data"]]
    assert any(token in t for t in titles)
    assert not any(f"Onsite Job {token}" == t for t in titles)


@pytest.mark.asyncio
async def test_salary_filter_excludes_undisclosed_salary(client: AsyncClient) -> None:
    token = _unique("sal")
    await _seed_job(
        title=f"Paid Job {token}",
        salary_min=90000,
        salary_max=110000,
        salary_period=SalaryPeriod.YEARLY,
    )
    await _seed_job(title=f"Undisclosed Job {token}")

    response = await client.get(
        "/api/v1/search", params={"salary_min": 50000, "salary_max": 150000, "page_size": 50}
    )
    assert response.status_code == 200
    titles = [item["title"] for item in response.json()["data"]]
    assert f"Paid Job {token}" in titles
    assert f"Undisclosed Job {token}" not in titles


@pytest.mark.asyncio
async def test_skills_filter_or_vs_and(client: AsyncClient) -> None:
    skill_a = _unique("skill-a-")
    skill_b = _unique("skill-b-")
    token = _unique("sk")
    await _seed_job(title=f"Both Skills {token}", skills=[skill_a, skill_b])
    await _seed_job(title=f"Only A {token}", skills=[skill_a])

    or_response = await client.get(
        "/api/v1/search", params={"skills": [skill_a, skill_b], "page_size": 50}
    )
    or_titles = [item["title"] for item in or_response.json()["data"]]
    assert f"Both Skills {token}" in or_titles
    assert f"Only A {token}" in or_titles

    and_response = await client.get(
        "/api/v1/search",
        params={"skills": [skill_a, skill_b], "skills_match_all": "true", "page_size": 50},
    )
    and_titles = [item["title"] for item in and_response.json()["data"]]
    assert f"Both Skills {token}" in and_titles
    assert f"Only A {token}" not in and_titles


@pytest.mark.asyncio
async def test_date_posted_filter(client: AsyncClient) -> None:
    token = _unique("dp")
    await _seed_job(title=f"Recent Job {token}", posted_at=datetime.now(UTC) - timedelta(hours=2))
    await _seed_job(title=f"Old Job {token}", posted_at=datetime.now(UTC) - timedelta(days=60))

    response = await client.get(
        "/api/v1/search", params={"date_posted": "last_24h", "page_size": 50}
    )
    titles = [item["title"] for item in response.json()["data"]]
    assert f"Recent Job {token}" in titles
    assert f"Old Job {token}" not in titles


@pytest.mark.asyncio
async def test_sort_by_newest(client: AsyncClient) -> None:
    token = _unique("sort-new")
    older = await _seed_job(title=f"Older {token}", posted_at=datetime.now(UTC) - timedelta(days=2))
    newer = await _seed_job(
        title=f"Newer {token}", posted_at=datetime.now(UTC) - timedelta(hours=1)
    )

    response = await client.get("/api/v1/search", params={"sort_by": "newest", "page_size": 50})
    ids = [item["id"] for item in response.json()["data"]]
    assert ids.index(str(newer.id)) < ids.index(str(older.id))


@pytest.mark.asyncio
async def test_sort_by_salary(client: AsyncClient) -> None:
    token = _unique("sort-sal")
    low = await _seed_job(
        title=f"Low Pay {token}",
        salary_min=40000,
        salary_max=50000,
        salary_period=SalaryPeriod.YEARLY,
    )
    high = await _seed_job(
        title=f"High Pay {token}",
        salary_min=180000,
        salary_max=200000,
        salary_period=SalaryPeriod.YEARLY,
    )

    response = await client.get("/api/v1/search", params={"sort_by": "salary", "page_size": 50})
    ids = [item["id"] for item in response.json()["data"]]
    assert ids.index(str(high.id)) < ids.index(str(low.id))


@pytest.mark.asyncio
async def test_autocomplete_min_chars_guard(client: AsyncClient) -> None:
    response = await client.get("/api/v1/search/autocomplete", params={"q": "a"})
    assert response.status_code == 200
    assert response.json()["data"] == []


@pytest.mark.asyncio
async def test_autocomplete_matches_title_prefix(client: AsyncClient) -> None:
    token = _unique("autocmp")
    await _seed_job(title=f"{token} Software Engineer")

    response = await client.get("/api/v1/search/autocomplete", params={"q": token})
    assert response.status_code == 200
    labels = [s["label"] for s in response.json()["data"]]
    assert any(token in label for label in labels)


@pytest.mark.asyncio
async def test_suggestions_returns_skill_and_title_types(client: AsyncClient) -> None:
    skill_name = _unique("suggest-skill-")
    await _seed_job(title=f"{skill_name} Engineer", skills=[skill_name])

    response = await client.get("/api/v1/search/suggestions", params={"q": skill_name})
    assert response.status_code == 200
    types = {s["type"] for s in response.json()["data"]}
    assert types <= {"skill", "job_title"}


@pytest.mark.asyncio
async def test_trending_skills_within_window(client: AsyncClient) -> None:
    skill_name = _unique("trend-skill-")
    await _seed_job(
        title=f"Trending Job {skill_name}",
        skills=[skill_name],
        posted_at=datetime.now(UTC) - timedelta(days=1),
    )

    response = await client.get(
        "/api/v1/search/trending/skills", params={"window_days": 30, "limit": 50}
    )
    assert response.status_code == 200
    names = [s["name"] for s in response.json()["data"]]
    assert skill_name in names


@pytest.mark.asyncio
async def test_trending_jobs_within_window(client: AsyncClient) -> None:
    token = _unique("trend-job")
    job = await _seed_job(
        title=f"Trending {token}", posted_at=datetime.now(UTC) - timedelta(hours=3)
    )

    response = await client.get(
        "/api/v1/search/trending/jobs", params={"window_days": 30, "limit": 50}
    )
    assert response.status_code == 200
    ids = [j["id"] for j in response.json()["data"]]
    assert str(job.id) in ids


@pytest.mark.asyncio
async def test_repeated_semantic_query_is_served_from_cache(
    client: AsyncClient, fake_embedding_provider: _FakeEmbeddingProvider
) -> None:
    token = _unique("cache")
    await _seed_job(title=f"Cached Search {token}", embedding=[0.1] * VECTOR_DIMENSIONS)

    params = {"q": token, "mode": "semantic"}
    first = await client.get("/api/v1/search", params=params)
    assert first.status_code == 200
    calls_after_first = len(fake_embedding_provider.calls)

    second = await client.get("/api/v1/search", params=params)
    assert second.status_code == 200
    assert len(fake_embedding_provider.calls) == calls_after_first
