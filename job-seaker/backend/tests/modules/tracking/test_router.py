"""Integration tests for the `/tracking` API — application CRUD, status
transitions, notes, timeline, and dashboard/statistics aggregates.

Needs a reachable Postgres with the tracking tables migrated:

    docker compose up -d postgres redis
    docker compose exec api alembic upgrade head
    docker compose exec api pytest tests/modules/tracking/test_router.py

Each test uses a freshly generated random `user_id` so tests don't need to
share or reset database state.
"""

from __future__ import annotations

import uuid

from httpx import AsyncClient

from app.core.security import create_access_token


def _user_header() -> dict[str, str]:
    token, _ = create_access_token(uuid.uuid4())
    return {"Authorization": f"Bearer {token}"}


async def _create_application(
    client: AsyncClient,
    headers: dict[str, str],
    *,
    company_name: str = "Acme Corp",
    job_title: str = "Backend Engineer",
    **overrides,
):
    payload = {"company_name": company_name, "job_title": job_title, **overrides}
    return await client.post("/api/v1/tracking/applications", headers=headers, json=payload)


async def test_create_application_records_initial_status_history(client: AsyncClient) -> None:
    headers = _user_header()
    response = await _create_application(client, headers)

    assert response.status_code == 201
    body = response.json()["data"]
    assert body["status"] == "applied"
    assert body["company_name"] == "Acme Corp"
    assert len(body["status_history"]) == 1
    assert body["status_history"][0]["from_status"] is None
    assert body["status_history"][0]["to_status"] == "applied"
    assert body["notes"] == []


async def test_endpoints_require_user_id_header(client: AsyncClient) -> None:
    response = await client.get("/api/v1/tracking/applications")
    assert response.status_code == 401


async def test_get_application_not_found_for_other_users(client: AsyncClient) -> None:
    headers_a = _user_header()
    headers_b = _user_header()
    create = await _create_application(client, headers_a)
    application_id = create.json()["data"]["id"]

    response = await client.get(
        f"/api/v1/tracking/applications/{application_id}", headers=headers_b
    )
    assert response.status_code == 404


async def test_list_applications_returns_only_the_caller_s_applications(
    client: AsyncClient,
) -> None:
    headers_a = _user_header()
    headers_b = _user_header()
    await _create_application(client, headers_a, company_name="A Corp")
    await _create_application(client, headers_b, company_name="B Corp")

    response = await client.get("/api/v1/tracking/applications", headers=headers_a)

    assert response.status_code == 200
    body = response.json()
    assert body["pagination"]["total_items"] == 1
    assert body["data"][0]["company_name"] == "A Corp"


async def test_filter_applications_by_status(client: AsyncClient) -> None:
    headers = _user_header()
    create = await _create_application(client, headers, company_name="Filtered Co")
    application_id = create.json()["data"]["id"]
    await client.post(
        f"/api/v1/tracking/applications/{application_id}/status",
        headers=headers,
        json={"status": "interview"},
    )
    await _create_application(client, headers, company_name="Still Applied Co")

    response = await client.get(
        "/api/v1/tracking/applications", headers=headers, params={"status": "interview"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["pagination"]["total_items"] == 1
    assert body["data"][0]["company_name"] == "Filtered Co"


async def test_search_applications_by_query(client: AsyncClient) -> None:
    headers = _user_header()
    await _create_application(client, headers, company_name="Searchable Widgets Inc")
    await _create_application(client, headers, company_name="Unrelated Co")

    response = await client.get(
        "/api/v1/tracking/applications", headers=headers, params={"q": "Searchable"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["pagination"]["total_items"] == 1
    assert body["data"][0]["company_name"] == "Searchable Widgets Inc"


async def test_sort_applications_by_company_name(client: AsyncClient) -> None:
    headers = _user_header()
    await _create_application(client, headers, company_name="Zeta Co")
    await _create_application(client, headers, company_name="Alpha Co")

    response = await client.get(
        "/api/v1/tracking/applications",
        headers=headers,
        params={"sort_by": "company_name", "sort_direction": "asc"},
    )

    assert response.status_code == 200
    names = [item["company_name"] for item in response.json()["data"]]
    assert names == sorted(names)


async def test_update_application_fields(client: AsyncClient) -> None:
    headers = _user_header()
    create = await _create_application(client, headers)
    application_id = create.json()["data"]["id"]

    response = await client.patch(
        f"/api/v1/tracking/applications/{application_id}",
        headers=headers,
        json={"job_title": "Senior Backend Engineer"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["job_title"] == "Senior Backend Engineer"


async def test_change_status_updates_status_and_records_history(client: AsyncClient) -> None:
    headers = _user_header()
    create = await _create_application(client, headers)
    application_id = create.json()["data"]["id"]

    response = await client.post(
        f"/api/v1/tracking/applications/{application_id}/status",
        headers=headers,
        json={"status": "interview", "note": "Recruiter screen scheduled."},
    )

    assert response.status_code == 200
    body = response.json()["data"]
    assert body["status"] == "interview"
    assert len(body["status_history"]) == 2
    assert body["status_history"][-1]["from_status"] == "applied"
    assert body["status_history"][-1]["to_status"] == "interview"
    assert body["status_history"][-1]["note"] == "Recruiter screen scheduled."

    history_response = await client.get(
        f"/api/v1/tracking/applications/{application_id}/status-history", headers=headers
    )
    assert history_response.status_code == 200
    assert len(history_response.json()["data"]) == 2


async def test_change_status_to_same_status_is_rejected(client: AsyncClient) -> None:
    headers = _user_header()
    create = await _create_application(client, headers)
    application_id = create.json()["data"]["id"]

    response = await client.post(
        f"/api/v1/tracking/applications/{application_id}/status",
        headers=headers,
        json={"status": "applied"},
    )

    assert response.status_code == 422


async def test_notes_crud_lifecycle(client: AsyncClient) -> None:
    headers = _user_header()
    create = await _create_application(client, headers)
    application_id = create.json()["data"]["id"]

    add_response = await client.post(
        f"/api/v1/tracking/applications/{application_id}/notes",
        headers=headers,
        json={"content": "Followed up via email."},
    )
    assert add_response.status_code == 201
    note_id = add_response.json()["data"]["id"]

    list_response = await client.get(
        f"/api/v1/tracking/applications/{application_id}/notes", headers=headers
    )
    assert list_response.status_code == 200
    assert len(list_response.json()["data"]) == 1

    update_response = await client.patch(
        f"/api/v1/tracking/applications/{application_id}/notes/{note_id}",
        headers=headers,
        json={"content": "Follow-up call scheduled for Friday."},
    )
    assert update_response.status_code == 200
    assert update_response.json()["data"]["content"] == "Follow-up call scheduled for Friday."

    delete_response = await client.delete(
        f"/api/v1/tracking/applications/{application_id}/notes/{note_id}", headers=headers
    )
    assert delete_response.status_code == 200

    final_list = await client.get(
        f"/api/v1/tracking/applications/{application_id}/notes", headers=headers
    )
    assert final_list.json()["data"] == []


async def test_timeline_merges_status_changes_and_notes_chronologically(
    client: AsyncClient,
) -> None:
    headers = _user_header()
    create = await _create_application(client, headers)
    application_id = create.json()["data"]["id"]

    await client.post(
        f"/api/v1/tracking/applications/{application_id}/notes",
        headers=headers,
        json={"content": "Applied via referral."},
    )
    await client.post(
        f"/api/v1/tracking/applications/{application_id}/status",
        headers=headers,
        json={"status": "interview"},
    )

    response = await client.get(
        f"/api/v1/tracking/applications/{application_id}/timeline", headers=headers
    )

    assert response.status_code == 200
    entries = response.json()["data"]
    assert len(entries) == 3  # initial status + note + status change
    types = [entry["type"] for entry in entries]
    assert types.count("status_change") == 2
    assert types.count("note") == 1
    timestamps = [entry["occurred_at"] for entry in entries]
    assert timestamps == sorted(timestamps)


async def test_delete_application_then_get_returns_404(client: AsyncClient) -> None:
    headers = _user_header()
    create = await _create_application(client, headers)
    application_id = create.json()["data"]["id"]

    delete_response = await client.delete(
        f"/api/v1/tracking/applications/{application_id}", headers=headers
    )
    assert delete_response.status_code == 200

    get_response = await client.get(
        f"/api/v1/tracking/applications/{application_id}", headers=headers
    )
    assert get_response.status_code == 404


async def test_statistics_reflect_status_counts(client: AsyncClient) -> None:
    headers = _user_header()
    create_a = await _create_application(client, headers, company_name="Stat Co A")
    create_b = await _create_application(client, headers, company_name="Stat Co B")
    application_id = create_a.json()["data"]["id"]
    await client.post(
        f"/api/v1/tracking/applications/{application_id}/status",
        headers=headers,
        json={"status": "offer"},
    )

    response = await client.get("/api/v1/tracking/statistics", headers=headers)

    assert response.status_code == 200
    body = response.json()["data"]
    assert body["total_applications"] == 2
    assert body["status_counts"]["offer"] == 1
    assert body["status_counts"]["applied"] == 1
    assert body["offer_rate"] == 0.5
    assert create_b.status_code == 201


async def test_dashboard_returns_statistics_and_recent_applications(
    client: AsyncClient,
) -> None:
    headers = _user_header()
    await _create_application(client, headers, company_name="Dashboard Co")

    response = await client.get("/api/v1/tracking/dashboard", headers=headers)

    assert response.status_code == 200
    body = response.json()["data"]
    assert body["statistics"]["total_applications"] == 1
    assert len(body["recent_applications"]) == 1
    assert body["recent_applications"][0]["company_name"] == "Dashboard Co"
