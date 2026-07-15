"""Integration tests for the `/ats-resumes` API — creation, versioning,
diffing, and download-link flows.

Needs a reachable Postgres (with resume_parser + job_ingestion +
ats_resume_generator tables migrated):

    docker compose up -d postgres redis
    docker compose exec api alembic upgrade head
    docker compose exec api pytest tests/modules/ats_resume_generator/test_router.py

The Claude optimize pass runs on a Celery worker in production; here
`optimize_ats_resume_task.delay` is faked to a no-op (same approach as
`resume_parser/test_router.py` faking `parse_resume_task.delay`), so these
tests exercise version/lifecycle bookkeeping without a running broker or
Anthropic API key. Each test seeds its own randomized Resume/ResumeVersion
directly through `AsyncSessionLocal` so tests don't need to share or reset
database state.
"""
from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient

from app.core.security import create_access_token
from app.db.session import AsyncSessionLocal
from app.modules.ats_resume_generator.tasks import optimize_ats_resume_task
from app.modules.resume_parser.models import (
    Resume,
    ResumeContentType,
    ResumeParsedData,
    ResumeVersion,
    ResumeVersionStatus,
)

JOB_DESCRIPTION = (
    "We are looking for a backend engineer with strong Python and PostgreSQL experience "
    "to join our platform team."
)


@pytest.fixture(autouse=True)
def _fake_external_dependencies(monkeypatch: pytest.MonkeyPatch):
    """No-op Celery enqueue — this suite only needs Postgres."""
    monkeypatch.setattr(optimize_ats_resume_task, "delay", lambda *a, **k: None)
    yield


def _user_header(user_id: uuid.UUID) -> dict[str, str]:
    token, _ = create_access_token(user_id)
    return {"Authorization": f"Bearer {token}"}


async def _seed_parsed_resume_version(
    *, user_id: uuid.UUID, summary: str = "Backend engineer.", skills: list[str] | None = None
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
                summary=summary,
                skills=skills or ["Python", "SQL"],
                projects=[
                    {
                        "name": "Order Service",
                        "description": "Built a payments pipeline.",
                        "technologies": ["Kafka"],
                        "highlights": ["Cut latency by 30%"],
                    }
                ],
                raw_text=f"{summary} Skilled in {', '.join(skills or ['Python', 'SQL'])}.",
                model_used="test-model",
            )
        )
        await session.commit()
        await session.refresh(version)
        return version


async def _seed_unparsed_resume_version(*, user_id: uuid.UUID) -> ResumeVersion:
    async with AsyncSessionLocal() as session:
        resume = Resume(user_id=user_id, title="My Resume")
        session.add(resume)
        await session.flush()

        version = ResumeVersion(
            resume_id=resume.id,
            version_number=1,
            is_current=True,
            status=ResumeVersionStatus.UPLOADED,
            content_type=ResumeContentType.PDF,
            original_filename="resume.pdf",
            s3_bucket="test-bucket",
            s3_key=f"resumes/{uuid.uuid4()}.pdf",
            file_size_bytes=1024,
        )
        session.add(version)
        await session.commit()
        await session.refresh(version)
        return version


async def _create_ats_resume(
    client: AsyncClient, headers: dict[str, str], resume_version_id: uuid.UUID, **overrides
):
    payload = {
        "resume_version_id": str(resume_version_id),
        "job_description": JOB_DESCRIPTION,
        "title": "Backend Engineer @ Acme",
        **overrides,
    }
    return await client.post("/api/v1/ats-resumes", headers=headers, json=payload)


async def test_create_ats_resume_creates_first_version(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    headers = _user_header(user_id)
    source = await _seed_parsed_resume_version(user_id=user_id)

    response = await _create_ats_resume(client, headers, source.id)

    assert response.status_code == 201
    body = response.json()["data"]
    assert body["title"] == "Backend Engineer @ Acme"
    version = body["current_version"]
    assert version["version_number"] == 1
    assert version["is_current"] is True
    assert version["status"] == "pending"


async def test_create_ats_resume_requires_parsed_source(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    headers = _user_header(user_id)
    source = await _seed_unparsed_resume_version(user_id=user_id)

    response = await _create_ats_resume(client, headers, source.id)

    assert response.status_code == 409


async def test_create_ats_resume_404_for_unowned_source(client: AsyncClient) -> None:
    owner_id = uuid.uuid4()
    other_id = uuid.uuid4()
    source = await _seed_parsed_resume_version(user_id=owner_id)

    response = await _create_ats_resume(client, _user_header(other_id), source.id)

    assert response.status_code == 404


async def test_endpoints_require_user_id_header(client: AsyncClient) -> None:
    response = await client.get("/api/v1/ats-resumes")
    assert response.status_code == 401


async def test_list_ats_resumes_returns_only_the_caller_s_resumes(client: AsyncClient) -> None:
    user_a, user_b = uuid.uuid4(), uuid.uuid4()
    source_a = await _seed_parsed_resume_version(user_id=user_a)
    source_b = await _seed_parsed_resume_version(user_id=user_b)
    await _create_ats_resume(client, _user_header(user_a), source_a.id, title="A's Resume")
    await _create_ats_resume(client, _user_header(user_b), source_b.id, title="B's Resume")

    response = await client.get("/api/v1/ats-resumes", headers=_user_header(user_a))

    assert response.status_code == 200
    body = response.json()
    assert body["pagination"]["total_items"] == 1
    assert body["data"][0]["title"] == "A's Resume"


async def test_get_ats_resume_not_found_for_other_users(client: AsyncClient) -> None:
    user_a, user_b = uuid.uuid4(), uuid.uuid4()
    source = await _seed_parsed_resume_version(user_id=user_a)
    created = await _create_ats_resume(client, _user_header(user_a), source.id)
    ats_resume_id = created.json()["data"]["id"]

    response = await client.get(
        f"/api/v1/ats-resumes/{ats_resume_id}", headers=_user_header(user_b)
    )
    assert response.status_code == 404


async def test_update_ats_resume_title(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    headers = _user_header(user_id)
    source = await _seed_parsed_resume_version(user_id=user_id)
    created = await _create_ats_resume(client, headers, source.id)
    ats_resume_id = created.json()["data"]["id"]

    response = await client.patch(
        f"/api/v1/ats-resumes/{ats_resume_id}", headers=headers, json={"title": "Renamed"}
    )

    assert response.status_code == 200
    assert response.json()["data"]["title"] == "Renamed"


async def test_create_new_version_reuses_job_description_when_omitted(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    headers = _user_header(user_id)
    source = await _seed_parsed_resume_version(user_id=user_id)
    created = await _create_ats_resume(client, headers, source.id)
    ats_resume_id = created.json()["data"]["id"]

    response = await client.post(
        f"/api/v1/ats-resumes/{ats_resume_id}/versions", headers=headers, json={}
    )

    assert response.status_code == 201
    body = response.json()["data"]
    assert body["version_number"] == 2
    assert body["is_current"] is True

    versions_response = await client.get(
        f"/api/v1/ats-resumes/{ats_resume_id}/versions", headers=headers
    )
    versions = versions_response.json()["data"]
    assert [v["version_number"] for v in versions] == [2, 1]

    detail_response = await client.get(
        f"/api/v1/ats-resumes/{ats_resume_id}/versions/{body['id']}", headers=headers
    )
    assert detail_response.json()["data"]["job_description"] == JOB_DESCRIPTION


async def test_activate_version_switches_current(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    headers = _user_header(user_id)
    source = await _seed_parsed_resume_version(user_id=user_id)
    created = await _create_ats_resume(client, headers, source.id)
    ats_resume_id = created.json()["data"]["id"]
    v1_id = created.json()["data"]["current_version"]["id"]

    await client.post(f"/api/v1/ats-resumes/{ats_resume_id}/versions", headers=headers, json={})

    response = await client.put(
        f"/api/v1/ats-resumes/{ats_resume_id}/versions/{v1_id}/activate", headers=headers
    )

    assert response.status_code == 200
    assert response.json()["data"]["is_current"] is True

    resume_response = await client.get(f"/api/v1/ats-resumes/{ats_resume_id}", headers=headers)
    assert resume_response.json()["data"]["current_version"]["id"] == v1_id


async def test_delete_only_version_is_rejected(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    headers = _user_header(user_id)
    source = await _seed_parsed_resume_version(user_id=user_id)
    created = await _create_ats_resume(client, headers, source.id)
    ats_resume_id = created.json()["data"]["id"]
    version_id = created.json()["data"]["current_version"]["id"]

    response = await client.delete(
        f"/api/v1/ats-resumes/{ats_resume_id}/versions/{version_id}", headers=headers
    )

    assert response.status_code == 409


async def test_delete_version_promotes_newest_remaining_to_current(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    headers = _user_header(user_id)
    source = await _seed_parsed_resume_version(user_id=user_id)
    created = await _create_ats_resume(client, headers, source.id)
    ats_resume_id = created.json()["data"]["id"]
    v1_id = created.json()["data"]["current_version"]["id"]

    v2 = await client.post(
        f"/api/v1/ats-resumes/{ats_resume_id}/versions", headers=headers, json={}
    )
    v2_id = v2.json()["data"]["id"]

    response = await client.delete(
        f"/api/v1/ats-resumes/{ats_resume_id}/versions/{v2_id}", headers=headers
    )
    assert response.status_code == 200

    resume_response = await client.get(f"/api/v1/ats-resumes/{ats_resume_id}", headers=headers)
    assert resume_response.json()["data"]["current_version"]["id"] == v1_id


async def test_delete_ats_resume_then_get_returns_404(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    headers = _user_header(user_id)
    source = await _seed_parsed_resume_version(user_id=user_id)
    created = await _create_ats_resume(client, headers, source.id)
    ats_resume_id = created.json()["data"]["id"]

    delete_response = await client.delete(f"/api/v1/ats-resumes/{ats_resume_id}", headers=headers)
    assert delete_response.status_code == 200

    get_response = await client.get(f"/api/v1/ats-resumes/{ats_resume_id}", headers=headers)
    assert get_response.status_code == 404


async def test_diff_against_source_before_optimization_shows_full_removal(
    client: AsyncClient,
) -> None:
    """Before the (faked/no-op) Celery optimize pass runs, `optimized_*` is
    still empty — diffing against 'source' should show the original summary
    fully removed and no additions, a deterministic way to exercise the diff
    endpoint without needing a real Claude call.
    """
    user_id = uuid.uuid4()
    headers = _user_header(user_id)
    source = await _seed_parsed_resume_version(user_id=user_id, summary="Backend engineer.")
    created = await _create_ats_resume(client, headers, source.id)
    ats_resume_id = created.json()["data"]["id"]
    version_id = created.json()["data"]["current_version"]["id"]

    response = await client.get(
        f"/api/v1/ats-resumes/{ats_resume_id}/versions/{version_id}/diff",
        headers=headers,
        params={"compare_to": "source"},
    )

    assert response.status_code == 200
    diff = response.json()["data"]
    assert diff["summary_diff"] == [{"op": "delete", "text": "Backend engineer."}]
    assert set(diff["skills_diff"]["removed"]) == {"Python", "SQL"}
    assert diff["skills_diff"]["added"] == []


async def test_diff_invalid_compare_to_returns_422(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    headers = _user_header(user_id)
    source = await _seed_parsed_resume_version(user_id=user_id)
    created = await _create_ats_resume(client, headers, source.id)
    ats_resume_id = created.json()["data"]["id"]
    version_id = created.json()["data"]["current_version"]["id"]

    response = await client.get(
        f"/api/v1/ats-resumes/{ats_resume_id}/versions/{version_id}/diff",
        headers=headers,
        params={"compare_to": "not-a-uuid"},
    )

    assert response.status_code == 422


async def test_download_404_before_files_generated(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    headers = _user_header(user_id)
    source = await _seed_parsed_resume_version(user_id=user_id)
    created = await _create_ats_resume(client, headers, source.id)
    ats_resume_id = created.json()["data"]["id"]
    version_id = created.json()["data"]["current_version"]["id"]

    response = await client.get(
        f"/api/v1/ats-resumes/{ats_resume_id}/versions/{version_id}/download/docx",
        headers=headers,
    )

    assert response.status_code == 404
