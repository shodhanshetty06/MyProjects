"""Integration tests for the `/analytics` API — dashboard aggregation and
CSV/PDF export, built on top of the `tracking` module's `Application` data.

Needs a reachable Postgres with the tracking tables migrated:

    docker compose up -d postgres redis
    docker compose exec api alembic upgrade head
    docker compose exec api pytest tests/modules/analytics/test_router.py

Each test uses a freshly generated random `user_id` so tests don't need to
share or reset database state.
"""

from __future__ import annotations

import uuid

from httpx import AsyncClient

from app.core.security import create_access_token


def _user_header() -> dict[str, str]:
    user_id = uuid.uuid4()
    token, _ = create_access_token(user_id)
    return {"X-User-Id": str(user_id), "Authorization": f"Bearer {token}"}


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


async def test_dashboard_requires_user_id_header(client: AsyncClient) -> None:
    response = await client.get("/api/v1/analytics/dashboard")
    assert response.status_code == 401


async def test_dashboard_reflects_the_caller_s_applications_only(client: AsyncClient) -> None:
    headers_a = _user_header()
    headers_b = _user_header()
    await _create_application(client, headers_a, company_name="A Corp")
    await _create_application(client, headers_b, company_name="B Corp")

    response = await client.get("/api/v1/analytics/dashboard", headers=headers_a)

    assert response.status_code == 200
    body = response.json()["data"]
    assert body["summary"]["total_applications"] == 1
    assert body["top_companies"][0]["company_name"] == "A Corp"


async def test_dashboard_offer_rate_and_acceptance_rate(client: AsyncClient) -> None:
    headers = _user_header()
    create = await _create_application(client, headers, company_name="Offer Co")
    application_id = create.json()["data"]["id"]
    await client.post(
        f"/api/v1/tracking/applications/{application_id}/status",
        headers=headers,
        json={"status": "offer"},
    )
    await client.post(
        f"/api/v1/tracking/applications/{application_id}/status",
        headers=headers,
        json={"status": "accepted"},
    )
    await _create_application(client, headers, company_name="Applied Only Co")

    response = await client.get("/api/v1/analytics/dashboard", headers=headers)

    assert response.status_code == 200
    summary = response.json()["data"]["summary"]
    assert summary["total_applications"] == 2
    assert summary["offer_rate"] == 0.5
    assert summary["acceptance_rate"] == 1.0


async def test_dashboard_applications_trend_has_no_gaps(client: AsyncClient) -> None:
    headers = _user_header()
    await _create_application(client, headers)

    response = await client.get(
        "/api/v1/analytics/dashboard", headers=headers, params={"interval": "day"}
    )

    assert response.status_code == 200
    points = response.json()["data"]["applications_trend"]["points"]
    assert len(points) > 0
    assert sum(p["count"] for p in points) == 1


async def test_dashboard_top_companies_tracks_interviews_and_offers(client: AsyncClient) -> None:
    headers = _user_header()
    create = await _create_application(client, headers, company_name="Interview Co")
    application_id = create.json()["data"]["id"]
    await client.post(
        f"/api/v1/tracking/applications/{application_id}/status",
        headers=headers,
        json={"status": "interview"},
    )

    response = await client.get("/api/v1/analytics/dashboard", headers=headers)

    assert response.status_code == 200
    company = response.json()["data"]["top_companies"][0]
    assert company["company_name"] == "Interview Co"
    assert company["applications"] == 1
    assert company["interviews"] == 1
    assert company["offers"] == 0


async def test_dashboard_heatmap_covers_the_requested_range(client: AsyncClient) -> None:
    headers = _user_header()
    await _create_application(client, headers)

    response = await client.get("/api/v1/analytics/dashboard", headers=headers)

    assert response.status_code == 200
    heatmap = response.json()["data"]["heatmap"]
    assert heatmap["total"] == 1
    assert len(heatmap["cells"]) > 0


async def test_export_csv_requires_user_id_header(client: AsyncClient) -> None:
    response = await client.get("/api/v1/analytics/export/csv")
    assert response.status_code == 401


async def test_export_csv_returns_applications_as_csv(client: AsyncClient) -> None:
    headers = _user_header()
    await _create_application(client, headers, company_name="CSV Export Co")

    response = await client.get("/api/v1/analytics/export/csv", headers=headers)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "CSV Export Co" in response.text


async def test_export_pdf_returns_a_pdf_document(client: AsyncClient) -> None:
    headers = _user_header()
    await _create_application(client, headers, company_name="PDF Export Co")

    response = await client.get("/api/v1/analytics/export/pdf", headers=headers)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")


async def test_dashboard_filters_by_date_range(client: AsyncClient) -> None:
    headers = _user_header()
    await _create_application(client, headers, company_name="In Range Co")

    response = await client.get(
        "/api/v1/analytics/dashboard",
        headers=headers,
        params={"date_from": "2000-01-01T00:00:00Z", "date_to": "2000-01-31T00:00:00Z"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["summary"]["total_applications"] == 0
