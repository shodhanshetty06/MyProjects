"""Integration tests for the Saved Jobs CRUD surface
(`POST/DELETE/GET /search/saved-jobs*`).

Needs a reachable Postgres (with job_ingestion + search's `saved_job` table
migrated):

    docker compose exec api alembic upgrade head
    docker compose exec api pytest tests/modules/search/test_saved_jobs.py

Redis publish (`search.realtime.publish_search_event`) is faked at the
function level (same approach as `matching/test_router.py` faking
`realtime.publish_match_event`), so no running Redis is required. Each test
seeds its own randomized Company/Job rows directly through
`AsyncSessionLocal`, mirroring `search/test_router.py`'s `_seed_job` helper.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient

from app.core.security import create_access_token
from app.db.session import AsyncSessionLocal
from app.modules.job_ingestion.models import ATSPlatform, Company, EmploymentType, Job, JobStatus, WorkMode
from app.modules.search import realtime


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


async def _seed_job() -> Job:
    async with AsyncSessionLocal() as session:
        company = Company(name=_unique("Company "), normalized_name=_unique("company-"))
        session.add(company)
        await session.flush()

        job = Job(
            company_id=company.id,
            source=ATSPlatform.CUSTOM_HTML,
            source_job_id=_unique("job-"),
            source_url="https://example.com/job",
            title=_unique("Engineer "),
            description="Build things.",
            employment_type=EmploymentType.FULL_TIME,
            work_mode=WorkMode.REMOTE,
            status=JobStatus.ACTIVE,
            content_hash=_unique("hash-"),
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return job


async def test_save_job_returns_201(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    job = await _seed_job()

    response = await client.post(
        f"/api/v1/search/saved-jobs/{job.id}", headers=_user_header(user_id)
    )
    assert response.status_code == 201
    body = response.json()["data"]
    assert body["job_id"] == str(job.id)


async def test_save_job_is_idempotent(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    job = await _seed_job()
    headers = _user_header(user_id)

    first = await client.post(f"/api/v1/search/saved-jobs/{job.id}", headers=headers)
    second = await client.post(f"/api/v1/search/saved-jobs/{job.id}", headers=headers)
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["data"]["id"] == second.json()["data"]["id"]


async def test_save_job_404_for_unknown_job(client: AsyncClient) -> None:
    response = await client.post(
        f"/api/v1/search/saved-jobs/{uuid.uuid4()}", headers=_user_header(uuid.uuid4())
    )
    assert response.status_code == 404


async def test_save_requires_authentication(client: AsyncClient) -> None:
    job = await _seed_job()
    response = await client.post(f"/api/v1/search/saved-jobs/{job.id}")
    assert response.status_code == 401


async def test_unsave_job(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    job = await _seed_job()
    headers = _user_header(user_id)

    await client.post(f"/api/v1/search/saved-jobs/{job.id}", headers=headers)
    delete_response = await client.delete(f"/api/v1/search/saved-jobs/{job.id}", headers=headers)
    assert delete_response.status_code == 200


async def test_unsave_job_404_when_not_saved(client: AsyncClient) -> None:
    job = await _seed_job()
    response = await client.delete(
        f"/api/v1/search/saved-jobs/{job.id}", headers=_user_header(uuid.uuid4())
    )
    assert response.status_code == 404


async def test_list_saved_jobs(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    job_a = await _seed_job()
    job_b = await _seed_job()
    headers = _user_header(user_id)

    await client.post(f"/api/v1/search/saved-jobs/{job_a.id}", headers=headers)
    await client.post(f"/api/v1/search/saved-jobs/{job_b.id}", headers=headers)

    response = await client.get("/api/v1/search/saved-jobs", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["pagination"]["total_items"] == 2
    ids = {item["id"] for item in body["data"]}
    assert ids == {str(job_a.id), str(job_b.id)}
    for item in body["data"]:
        assert item["is_saved"] is True
        assert "ai_signals" in item


async def test_saved_jobs_scoped_per_user(client: AsyncClient) -> None:
    job = await _seed_job()
    owner_headers = _user_header(uuid.uuid4())
    other_headers = _user_header(uuid.uuid4())

    await client.post(f"/api/v1/search/saved-jobs/{job.id}", headers=owner_headers)

    other_list = await client.get("/api/v1/search/saved-jobs", headers=other_headers)
    assert other_list.json()["pagination"]["total_items"] == 0
