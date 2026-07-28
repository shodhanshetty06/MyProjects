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


# --- Job Search integration (Section 8) -----------------------------------------------


async def test_dashboard_saved_jobs_count_defaults_to_zero(client: AsyncClient) -> None:
    headers = _user_header()
    response = await client.get("/api/v1/analytics/dashboard", headers=headers)
    assert response.status_code == 200
    body = response.json()["data"]
    assert body["saved_jobs_count"] == 0
    assert body["average_match_score"] is None
    assert body["top_matched_companies"] == []


async def test_dashboard_saved_jobs_count_reflects_saved_job_table(client: AsyncClient) -> None:
    import uuid

    from app.db.session import AsyncSessionLocal
    from app.modules.job_ingestion.models import ATSPlatform, Company, EmploymentType, Job, JobStatus, WorkMode

    user_id = uuid.uuid4()
    token, _ = create_access_token(user_id)
    headers = {"Authorization": f"Bearer {token}"}

    async with AsyncSessionLocal() as session:
        company = Company(name="Save Co", normalized_name=f"save-co-{uuid.uuid4()}")
        session.add(company)
        await session.flush()
        job = Job(
            company_id=company.id,
            source=ATSPlatform.CUSTOM_HTML,
            source_job_id=str(uuid.uuid4()),
            source_url="https://example.com/job",
            title="Engineer",
            description="Build things.",
            employment_type=EmploymentType.FULL_TIME,
            work_mode=WorkMode.REMOTE,
            status=JobStatus.ACTIVE,
            content_hash=str(uuid.uuid4()),
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)

    save_response = await client.post(f"/api/v1/search/saved-jobs/{job.id}", headers=headers)
    assert save_response.status_code == 201

    response = await client.get("/api/v1/analytics/dashboard", headers=headers)
    assert response.status_code == 200
    assert response.json()["data"]["saved_jobs_count"] == 1


async def test_dashboard_match_summary_computed_from_matching_repository(
    client: AsyncClient,
) -> None:
    """`average_match_score`/`top_matched_companies` are sourced from
    `MatchingRepository.list_matches_for_user` over a directly-seeded
    `ResumeJobMatch` row — proves the dashboard reads real match data
    rather than reimplementing scoring.
    """
    import uuid

    from app.db.session import AsyncSessionLocal
    from app.modules.job_ingestion.models import ATSPlatform, Company, EmploymentType, Job, JobStatus, WorkMode
    from app.modules.matching.models import MatchStatus, ResumeJobMatch
    from app.modules.resume_parser.models import Resume, ResumeContentType, ResumeVersion, ResumeVersionStatus

    user_id = uuid.uuid4()
    token, _ = create_access_token(user_id)
    headers = {"Authorization": f"Bearer {token}"}

    async with AsyncSessionLocal() as session:
        company = Company(name="Matched Co", normalized_name=f"matched-co-{uuid.uuid4()}")
        session.add(company)
        await session.flush()
        job = Job(
            company_id=company.id,
            source=ATSPlatform.CUSTOM_HTML,
            source_job_id=str(uuid.uuid4()),
            source_url="https://example.com/job",
            title="Engineer",
            description="Build things.",
            employment_type=EmploymentType.FULL_TIME,
            work_mode=WorkMode.REMOTE,
            status=JobStatus.ACTIVE,
            content_hash=str(uuid.uuid4()),
        )
        session.add(job)
        await session.flush()

        resume = Resume(user_id=user_id, title="R")
        session.add(resume)
        await session.flush()
        version = ResumeVersion(
            resume_id=resume.id,
            version_number=1,
            is_current=True,
            status=ResumeVersionStatus.PARSED,
            content_type=ResumeContentType.PDF,
            original_filename="r.pdf",
            s3_bucket="b",
            s3_key=f"k-{uuid.uuid4()}",
            file_size_bytes=1,
        )
        session.add(version)
        await session.flush()

        match = ResumeJobMatch(
            resume_version_id=version.id,
            job_id=job.id,
            user_id=user_id,
            status=MatchStatus.COMPUTED,
            match_score=88.0,
            matched_skills=[],
            missing_skills=[],
            strengths=[],
            weaknesses=[],
            score_breakdown={},
        )
        session.add(match)
        await session.commit()

    response = await client.get("/api/v1/analytics/dashboard", headers=headers)
    assert response.status_code == 200
    body = response.json()["data"]
    assert body["average_match_score"] == 88.0
    assert body["top_matched_companies"][0]["company_name"] == "Matched Co"
    assert body["top_matched_companies"][0]["matches"] == 1
