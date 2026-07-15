"""Integration tests for the `/job-ingestion` API — company registration,
crawl triggering, and job listing.

Needs a reachable Postgres with the job_ingestion tables migrated:

    docker compose up -d postgres redis
    docker compose exec api alembic upgrade head
    docker compose exec api pytest tests/modules/job_ingestion/test_router.py

Celery is faked at the task level so no running broker is required. Each
test registers a company with a randomized name so tests don't need to
share or reset database state.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient

from app.modules.job_ingestion.tasks import fetch_company_jobs_task


@pytest.fixture(autouse=True)
def _fake_celery(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(fetch_company_jobs_task, "delay", lambda *args, **kwargs: None)
    yield


def _unique_name() -> str:
    return f"Acme Corp {uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio
async def test_register_and_get_company(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/job-ingestion/companies",
        json={
            "name": _unique_name(),
            "website": "https://acme.example.com",
            "intelligence": {"ats_platform": "greenhouse", "ats_identifier": "acme"},
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    company_id = body["data"]["id"]
    assert body["data"]["intelligence"]["ats_platform"] == "greenhouse"

    get_response = await client.get(f"/api/v1/job-ingestion/companies/{company_id}")
    assert get_response.status_code == 200
    assert get_response.json()["data"]["id"] == company_id


@pytest.mark.asyncio
async def test_register_company_duplicate_name_conflicts(client: AsyncClient) -> None:
    name = _unique_name()
    first = await client.post("/api/v1/job-ingestion/companies", json={"name": name})
    assert first.status_code == 201

    second = await client.post("/api/v1/job-ingestion/companies", json={"name": name})
    assert second.status_code == 409


@pytest.mark.asyncio
async def test_list_companies_paginated(client: AsyncClient) -> None:
    await client.post("/api/v1/job-ingestion/companies", json={"name": _unique_name()})

    response = await client.get(
        "/api/v1/job-ingestion/companies", params={"page": 1, "page_size": 5}
    )
    assert response.status_code == 200
    body = response.json()
    assert "pagination" in body
    assert body["pagination"]["page"] == 1


@pytest.mark.asyncio
async def test_update_company_intelligence(client: AsyncClient) -> None:
    create = await client.post("/api/v1/job-ingestion/companies", json={"name": _unique_name()})
    company_id = create.json()["data"]["id"]

    response = await client.patch(
        f"/api/v1/job-ingestion/companies/{company_id}",
        json={"intelligence": {"ats_platform": "lever", "ats_identifier": "acme"}},
    )
    assert response.status_code == 200
    assert response.json()["data"]["intelligence"]["ats_platform"] == "lever"


@pytest.mark.asyncio
async def test_trigger_crawl_requires_ats_platform_configured(client: AsyncClient) -> None:
    create = await client.post("/api/v1/job-ingestion/companies", json={"name": _unique_name()})
    company_id = create.json()["data"]["id"]

    response = await client.post(f"/api/v1/job-ingestion/companies/{company_id}/crawl")
    assert response.status_code == 422
    assert response.json()["error"]["error_code"] == "ats_platform_not_configured"


@pytest.mark.asyncio
async def test_trigger_crawl_succeeds_once_configured(client: AsyncClient) -> None:
    create = await client.post(
        "/api/v1/job-ingestion/companies",
        json={
            "name": _unique_name(),
            "intelligence": {"ats_platform": "greenhouse", "ats_identifier": "acme"},
        },
    )
    company_id = create.json()["data"]["id"]

    response = await client.post(f"/api/v1/job-ingestion/companies/{company_id}/crawl")
    assert response.status_code == 202


@pytest.mark.asyncio
async def test_list_jobs_empty(client: AsyncClient) -> None:
    response = await client.get("/api/v1/job-ingestion/jobs")
    assert response.status_code == 200
    assert response.json()["pagination"]["total_items"] >= 0


@pytest.mark.asyncio
async def test_get_company_not_found(client: AsyncClient) -> None:
    response = await client.get(f"/api/v1/job-ingestion/companies/{uuid.uuid4()}")
    assert response.status_code == 404
