"""Integration tests for the `/admin` API — bootstrap, RBAC enforcement,
user/company/job moderation, reports, feature flags, and audit logging.

Needs a reachable Postgres with the admin tables migrated:

    docker compose up -d postgres redis
    docker compose exec api alembic upgrade head
    docker compose exec api pytest tests/modules/admin/test_router.py

Each test uses a freshly generated random `user_id` per actor so tests don't
need to share or reset database state — except bootstrap, which is
process-global (only the *first* admin_actor row ever created can bootstrap
as SUPER_ADMIN); `test_bootstrap_only_works_once` accounts for that by not
assuming it is the first bootstrap call in the suite.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.core.security import create_access_token
from app.db.session import AsyncSessionLocal
from app.modules.job_ingestion.models import Job


def _user_header() -> dict[str, str]:
    """`X-User-Id` is inert bookkeeping now (the backend authenticates off
    `Authorization` only) — kept so call sites can still read the raw id
    they authenticated as via `headers['X-User-Id']`."""
    user_id = uuid.uuid4()
    token, _ = create_access_token(user_id)
    return {"X-User-Id": str(user_id), "Authorization": f"Bearer {token}"}


async def _bootstrap_super_admin(client: AsyncClient) -> dict[str, str]:
    """Returns headers for *some* SUPER_ADMIN/ADMIN actor.

    Bootstrap only succeeds once process-wide, so if it's already taken this
    creates a fresh SUPER_ADMIN through an existing one instead (requires at
    least one prior successful bootstrap in this test session, which the
    first call to this helper always performs).
    """
    headers = _user_header()
    response = await client.post(
        "/api/v1/admin/actors/bootstrap",
        headers=headers,
        json={"email": f"{headers['X-User-Id']}@example.com", "display_name": "Root Admin"},
    )
    if response.status_code == 201:
        return headers

    # Bootstrap already used by another test — fall back to a still-known
    # SUPER_ADMIN captured on the module the first time this ran.
    assert _KNOWN_SUPER_ADMIN["headers"] is not None
    return _KNOWN_SUPER_ADMIN["headers"]


_KNOWN_SUPER_ADMIN: dict[str, dict[str, str] | None] = {"headers": None}


@pytest.fixture
async def super_admin_headers(client: AsyncClient) -> dict[str, str]:
    if _KNOWN_SUPER_ADMIN["headers"] is not None:
        return _KNOWN_SUPER_ADMIN["headers"]
    headers = await _bootstrap_super_admin(client)
    _KNOWN_SUPER_ADMIN["headers"] = headers
    return headers


async def _seed_job(company_id: uuid.UUID) -> uuid.UUID:
    async with AsyncSessionLocal() as session:
        job = Job(
            company_id=company_id,
            source="custom_html",
            source_job_id=uuid.uuid4().hex,
            source_url="https://example.com/job",
            title="Senior Backend Engineer",
            description="Build things.",
            content_hash=uuid.uuid4().hex,
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return job.id


async def _register_company(client: AsyncClient) -> uuid.UUID:
    name = f"Acme Corp {uuid.uuid4().hex[:8]}"
    response = await client.post("/api/v1/job-ingestion/companies", json={"name": name})
    assert response.status_code == 201
    return uuid.UUID(response.json()["data"]["id"])


# --- auth / permission enforcement ------------------------------------------------


async def test_dashboard_requires_user_id_header(client: AsyncClient) -> None:
    response = await client.get("/api/v1/admin/dashboard")
    assert response.status_code == 401


async def test_dashboard_forbidden_for_unregistered_actor(client: AsyncClient) -> None:
    response = await client.get("/api/v1/admin/dashboard", headers=_user_header())
    assert response.status_code == 403


async def test_role_without_permission_is_forbidden(
    client: AsyncClient, super_admin_headers: dict[str, str]
) -> None:
    support_user = _user_header()
    create = await client.post(
        "/api/v1/admin/actors",
        headers=super_admin_headers,
        json={
            "user_id": support_user["X-User-Id"],
            "email": "support@example.com",
            "display_name": "Support Agent",
            "role": "support",
        },
    )
    assert create.status_code == 201

    # SUPPORT has USERS_MANAGE + DASHBOARD_VIEW (broad read, incl. GET
    # /companies) but not COMPANIES_VERIFY — only the verification mutation
    # is gated.
    company_id = await _register_company(client)
    response = await client.patch(
        f"/api/v1/admin/companies/{company_id}/verification",
        headers=support_user,
        json={"status": "verified"},
    )
    assert response.status_code == 403

    companies_response = await client.get("/api/v1/admin/companies", headers=support_user)
    assert companies_response.status_code == 200

    ok_response = await client.get("/api/v1/admin/dashboard", headers=support_user)
    assert ok_response.status_code == 200


# --- bootstrap ----------------------------------------------------------------


async def test_bootstrap_only_works_once(
    client: AsyncClient, super_admin_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/admin/actors/bootstrap",
        headers=_user_header(),
        json={"email": "someone-else@example.com"},
    )
    assert response.status_code == 409


# --- permissions ----------------------------------------------------------------


async def test_get_permissions_reflects_super_admin_role(
    client: AsyncClient, super_admin_headers: dict[str, str]
) -> None:
    response = await client.get("/api/v1/admin/permissions", headers=super_admin_headers)
    assert response.status_code == 200
    body = response.json()["data"]
    assert body["current_role"] == "super_admin"
    assert "permissions.manage" in body["current_permissions"]
    assert len(body["roles"]) == 5


# --- feature flags ----------------------------------------------------------------


async def test_feature_flag_crud_and_audit_trail(
    client: AsyncClient, super_admin_headers: dict[str, str]
) -> None:
    key = f"new-dashboard-{uuid.uuid4().hex[:8]}"

    create = await client.post(
        "/api/v1/admin/feature-flags",
        headers=super_admin_headers,
        json={"key": key, "name": "New Dashboard", "is_enabled": False},
    )
    assert create.status_code == 201
    assert create.json()["data"]["is_enabled"] is False

    duplicate = await client.post(
        "/api/v1/admin/feature-flags",
        headers=super_admin_headers,
        json={"key": key, "name": "Dup"},
    )
    assert duplicate.status_code == 409

    update = await client.patch(
        f"/api/v1/admin/feature-flags/{key}",
        headers=super_admin_headers,
        json={"is_enabled": True},
    )
    assert update.status_code == 200
    assert update.json()["data"]["is_enabled"] is True

    listed = await client.get("/api/v1/admin/feature-flags", headers=super_admin_headers)
    assert any(flag["key"] == key for flag in listed.json()["data"])

    audit = await client.get(
        "/api/v1/admin/audit-logs",
        headers=super_admin_headers,
        params={"target_type": "feature_flag", "page_size": 50},
    )
    actions = [entry["action"] for entry in audit.json()["data"]]
    assert "feature_flag.create" in actions
    assert "feature_flag.update" in actions

    delete = await client.delete(f"/api/v1/admin/feature-flags/{key}", headers=super_admin_headers)
    assert delete.status_code == 200


# --- users ----------------------------------------------------------------------


async def test_user_status_update_flow(
    client: AsyncClient, super_admin_headers: dict[str, str]
) -> None:
    target_user = _user_header()
    await client.post(
        "/api/v1/tracking/applications",
        headers=target_user,
        json={"company_name": "Acme", "job_title": "Engineer"},
    )

    listed = await client.get(
        "/api/v1/admin/users",
        headers=super_admin_headers,
        params={"search": target_user["X-User-Id"]},
    )
    assert listed.status_code == 200
    assert listed.json()["data"][0]["user_id"] == target_user["X-User-Id"]

    detail = await client.get(
        f"/api/v1/admin/users/{target_user['X-User-Id']}", headers=super_admin_headers
    )
    assert detail.status_code == 200
    assert detail.json()["data"]["status"] == "active"

    suspend = await client.patch(
        f"/api/v1/admin/users/{target_user['X-User-Id']}/status",
        headers=super_admin_headers,
        json={"status": "suspended", "reason": "Spam applications"},
    )
    assert suspend.status_code == 200
    assert suspend.json()["data"]["status"] == "suspended"
    assert suspend.json()["data"]["reason"] == "Spam applications"


# --- companies --------------------------------------------------------------------


async def test_company_verification_flow(
    client: AsyncClient, super_admin_headers: dict[str, str]
) -> None:
    company_id = await _register_company(client)

    detail = await client.get(f"/api/v1/admin/companies/{company_id}", headers=super_admin_headers)
    assert detail.status_code == 200
    assert detail.json()["data"]["verification_status"] == "unverified"

    verify = await client.patch(
        f"/api/v1/admin/companies/{company_id}/verification",
        headers=super_admin_headers,
        json={"status": "verified", "notes": "Looks legitimate"},
    )
    assert verify.status_code == 200
    assert verify.json()["data"]["verification_status"] == "verified"

    listed = await client.get(
        "/api/v1/admin/companies",
        headers=super_admin_headers,
        params={"verification_status": "verified"},
    )
    assert any(c["id"] == str(company_id) for c in listed.json()["data"])


# --- jobs -----------------------------------------------------------------------


async def test_job_moderation_soft_deletes_removed_job(
    client: AsyncClient, super_admin_headers: dict[str, str]
) -> None:
    company_id = await _register_company(client)
    job_id = await _seed_job(company_id)

    flag = await client.patch(
        f"/api/v1/admin/jobs/{job_id}/moderation",
        headers=super_admin_headers,
        json={"status": "removed", "reason": "Fraudulent posting"},
    )
    assert flag.status_code == 200
    assert flag.json()["data"]["moderation_status"] == "removed"

    async with AsyncSessionLocal() as session:
        job = await session.get(Job, job_id)
        assert job.deleted_at is not None


# --- reports / content moderation --------------------------------------------------


async def test_report_lifecycle_and_moderation_queue(
    client: AsyncClient, super_admin_headers: dict[str, str]
) -> None:
    reporter = _user_header()
    target_id = uuid.uuid4()

    create = await client.post(
        "/api/v1/admin/reports",
        headers=reporter,
        json={"target_type": "job", "target_id": str(target_id), "reason": "Looks like a scam"},
    )
    assert create.status_code == 201
    report_id = create.json()["data"]["id"]
    assert create.json()["data"]["status"] == "open"

    queue = await client.get("/api/v1/admin/moderation/queue", headers=super_admin_headers)
    assert queue.status_code == 200
    assert any(item["id"] == report_id for item in queue.json()["data"]["items"])

    resolve = await client.patch(
        f"/api/v1/admin/reports/{report_id}",
        headers=super_admin_headers,
        json={"status": "resolved", "resolution_notes": "Confirmed and removed"},
    )
    assert resolve.status_code == 200
    assert resolve.json()["data"]["status"] == "resolved"


# --- read-only aggregates ----------------------------------------------------------


async def test_dashboard_system_logs_and_analytics_are_reachable(
    client: AsyncClient, super_admin_headers: dict[str, str]
) -> None:
    dashboard = await client.get("/api/v1/admin/dashboard", headers=super_admin_headers)
    assert dashboard.status_code == 200
    assert "generated_at" in dashboard.json()["data"]

    system_logs = await client.get("/api/v1/admin/system-logs", headers=super_admin_headers)
    assert system_logs.status_code == 200

    analytics = await client.get(
        "/api/v1/admin/analytics", headers=super_admin_headers, params={"days": 7}
    )
    assert analytics.status_code == 200
    assert analytics.json()["data"]["days"] == 7
