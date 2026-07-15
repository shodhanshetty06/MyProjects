"""Integration tests for the `/auto-apply` API — applicant profile CRUD and
the auto-apply task lifecycle's contract-level behavior (not-found/conflict
handling). Full happy-path task submission is exercised by Celery workers
against real ATS pages, not by this HTTP-only test suite.

Needs a reachable Postgres with every module's tables migrated:

    docker compose up -d postgres redis
    docker compose exec api alembic upgrade head
    docker compose exec api pytest tests/modules/auto_apply/test_router.py
"""
from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient

from app.core.security import create_access_token
from app.modules.auto_apply.tasks import submit_application_task


@pytest.fixture(autouse=True)
def _fake_celery(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(submit_application_task, "delay", lambda *args, **kwargs: None)
    yield


def _headers() -> dict[str, str]:
    token, _ = create_access_token(uuid.uuid4())
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_get_profile_creates_an_empty_profile_on_first_access(client: AsyncClient) -> None:
    headers = _headers()
    response = await client.get("/api/v1/auto-apply/profile", headers=headers)
    assert response.status_code == 200
    body = response.json()["data"]
    assert body["first_name"] is None
    assert body["auto_apply_enabled"] is True


@pytest.mark.asyncio
async def test_upsert_profile_round_trips(client: AsyncClient) -> None:
    headers = _headers()
    payload = {
        "first_name": "Ada",
        "last_name": "Lovelace",
        "email": "ada@example.com",
        "phone": "+14155550100",
        "work_authorized": True,
        "requires_sponsorship": False,
        "custom_answers": {"Why us?": "Because I love the mission."},
    }
    response = await client.put("/api/v1/auto-apply/profile", json=payload, headers=headers)
    assert response.status_code == 200
    body = response.json()["data"]
    assert body["first_name"] == "Ada"
    assert body["custom_answers"] == {"Why us?": "Because I love the mission."}

    get_response = await client.get("/api/v1/auto-apply/profile", headers=headers)
    assert get_response.json()["data"]["email"] == "ada@example.com"


@pytest.mark.asyncio
async def test_profile_is_scoped_per_user(client: AsyncClient) -> None:
    await client.put(
        "/api/v1/auto-apply/profile", json={"first_name": "Ada"}, headers=_headers()
    )
    other_response = await client.get("/api/v1/auto-apply/profile", headers=_headers())
    assert other_response.json()["data"]["first_name"] is None


@pytest.mark.asyncio
async def test_create_task_for_unknown_job_is_not_found(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/auto-apply/tasks", json={"job_id": str(uuid.uuid4())}, headers=_headers()
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_unknown_task_is_not_found(client: AsyncClient) -> None:
    response = await client.get(f"/api/v1/auto-apply/tasks/{uuid.uuid4()}", headers=_headers())
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_retry_unknown_task_is_not_found(client: AsyncClient) -> None:
    response = await client.post(
        f"/api/v1/auto-apply/tasks/{uuid.uuid4()}/retry", headers=_headers()
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_cancel_unknown_task_is_not_found(client: AsyncClient) -> None:
    response = await client.post(
        f"/api/v1/auto-apply/tasks/{uuid.uuid4()}/cancel", headers=_headers()
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_tasks_empty_for_new_user(client: AsyncClient) -> None:
    response = await client.get("/api/v1/auto-apply/tasks", headers=_headers())
    assert response.status_code == 200
    assert response.json()["data"] == []


@pytest.mark.asyncio
async def test_requests_without_user_header_are_unauthorized(client: AsyncClient) -> None:
    response = await client.get("/api/v1/auto-apply/profile")
    assert response.status_code == 401
