"""Integration tests for the `/resumes` API — upload, list, versioning,
activation, and delete flows.

Unlike `test_extraction.py` / `test_ai_service.py` (pure unit tests), these
exercise the real FastAPI app end-to-end and therefore need a reachable
Postgres with the resume_parser tables migrated:

    docker compose up -d postgres redis
    docker compose exec api alembic upgrade head
    docker compose exec api pytest tests/modules/resume_parser/test_router.py

S3 and Celery are faked at the dependency/task level so no AWS credentials
or running broker are required. Each test uses a freshly generated random
`user_id` so tests don't need to share or reset database state.
"""
from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient

from app.api.deps import get_storage_provider
from app.core.security import create_access_token
from app.main import app
from app.modules.resume_parser.tasks import parse_resume_task

PDF_BYTES = b"%PDF-1.4\n%fake-but-good-enough-for-content-type-validation\n%%EOF"
DOCX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class _FakeStorageProvider:
    """In-memory stand-in for a `StorageProvider` — no cloud credentials required."""

    def __init__(self) -> None:
        self.bucket = "test-bucket"
        self._objects: dict[str, bytes] = {}

    def upload_bytes(self, data: bytes, key: str) -> None:
        self._objects[key] = data

    def download_bytes(self, key: str) -> bytes:
        return self._objects[key]

    def delete_object(self, key: str) -> None:
        self._objects.pop(key, None)

    def get_download_url(self, key: str, expires_in: int = 3600) -> str:
        return f"https://example-bucket.s3.amazonaws.com/{key}?X-Fake-Signature=1"


@pytest.fixture(autouse=True)
def _fake_external_dependencies(monkeypatch: pytest.MonkeyPatch):
    """Replace storage with an in-memory fake and no-op the parsing Celery task.

    Without this, `upload_resume` would try to reach a real storage backend
    and a real Redis-backed broker.
    """
    app.dependency_overrides[get_storage_provider] = lambda: _FakeStorageProvider()
    monkeypatch.setattr(parse_resume_task, "delay", lambda *args, **kwargs: None)
    yield
    app.dependency_overrides.pop(get_storage_provider, None)


def _user_header() -> dict[str, str]:
    token, _ = create_access_token(uuid.uuid4())
    return {"Authorization": f"Bearer {token}"}


async def _upload_resume(client: AsyncClient, headers: dict[str, str], title: str = "My Resume"):
    files = {"file": ("resume.pdf", PDF_BYTES, "application/pdf")}
    return await client.post(
        "/api/v1/resumes", headers=headers, files=files, data={"title": title}
    )


async def test_upload_resume_creates_resume_and_first_version(client: AsyncClient) -> None:
    headers = _user_header()
    response = await _upload_resume(client, headers, title="Backend Engineer Resume")

    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    assert body["data"]["title"] == "Backend Engineer Resume"
    version = body["data"]["current_version"]
    assert version["version_number"] == 1
    assert version["is_current"] is True
    assert version["status"] == "uploaded"
    assert version["content_type"] == "pdf"


async def test_upload_resume_rejects_unsupported_file_type(client: AsyncClient) -> None:
    headers = _user_header()
    files = {"file": ("resume.txt", b"plain text resume", "text/plain")}

    response = await client.post("/api/v1/resumes", headers=headers, files=files)

    assert response.status_code == 422
    assert response.json()["error"]["error_code"] == "unsupported_file_type"


async def test_upload_resume_rejects_oversized_file(client: AsyncClient) -> None:
    headers = _user_header()
    oversized = b"0" * (11 * 1024 * 1024)  # exceeds the 10MB default limit
    files = {"file": ("resume.pdf", oversized, "application/pdf")}

    response = await client.post("/api/v1/resumes", headers=headers, files=files)

    assert response.status_code == 422
    assert response.json()["error"]["error_code"] == "file_too_large"


async def test_endpoints_require_user_id_header(client: AsyncClient) -> None:
    response = await client.get("/api/v1/resumes")
    assert response.status_code == 401


async def test_list_resumes_returns_only_the_caller_s_resumes(client: AsyncClient) -> None:
    headers_a = _user_header()
    headers_b = _user_header()
    await _upload_resume(client, headers_a, title="A's Resume")
    await _upload_resume(client, headers_b, title="B's Resume")

    response = await client.get("/api/v1/resumes", headers=headers_a)

    assert response.status_code == 200
    body = response.json()
    assert body["pagination"]["total_items"] == 1
    assert body["data"][0]["title"] == "A's Resume"


async def test_get_resume_not_found_for_other_users(client: AsyncClient) -> None:
    headers_a = _user_header()
    headers_b = _user_header()
    upload = await _upload_resume(client, headers_a)
    resume_id = upload.json()["data"]["id"]

    response = await client.get(f"/api/v1/resumes/{resume_id}", headers=headers_b)

    assert response.status_code == 404


async def test_update_resume_title(client: AsyncClient) -> None:
    headers = _user_header()
    upload = await _upload_resume(client, headers)
    resume_id = upload.json()["data"]["id"]

    response = await client.patch(
        f"/api/v1/resumes/{resume_id}", headers=headers, json={"title": "Renamed Resume"}
    )

    assert response.status_code == 200
    assert response.json()["data"]["title"] == "Renamed Resume"


async def test_upload_new_version_becomes_current(client: AsyncClient) -> None:
    headers = _user_header()
    upload = await _upload_resume(client, headers)
    resume_id = upload.json()["data"]["id"]

    files = {"file": ("resume_v2.pdf", PDF_BYTES, "application/pdf")}
    response = await client.post(
        f"/api/v1/resumes/{resume_id}/versions", headers=headers, files=files
    )

    assert response.status_code == 201
    body = response.json()["data"]
    assert body["version_number"] == 2
    assert body["is_current"] is True

    versions_response = await client.get(f"/api/v1/resumes/{resume_id}/versions", headers=headers)
    versions = versions_response.json()["data"]
    assert [v["version_number"] for v in versions] == [2, 1]


async def test_get_version_detail_before_parsing_has_no_parsed_data(client: AsyncClient) -> None:
    headers = _user_header()
    upload = await _upload_resume(client, headers)
    resume_id = upload.json()["data"]["id"]
    version_id = upload.json()["data"]["current_version"]["id"]

    response = await client.get(
        f"/api/v1/resumes/{resume_id}/versions/{version_id}", headers=headers
    )

    assert response.status_code == 200
    body = response.json()["data"]
    assert body["parsed_data"] is None
    assert body["has_embedding"] is False


async def test_activate_version_switches_current(client: AsyncClient) -> None:
    headers = _user_header()
    upload = await _upload_resume(client, headers)
    resume_id = upload.json()["data"]["id"]
    v1_id = upload.json()["data"]["current_version"]["id"]

    files = {"file": ("resume_v2.pdf", PDF_BYTES, "application/pdf")}
    await client.post(f"/api/v1/resumes/{resume_id}/versions", headers=headers, files=files)

    response = await client.put(
        f"/api/v1/resumes/{resume_id}/versions/{v1_id}/activate", headers=headers
    )

    assert response.status_code == 200
    assert response.json()["data"]["is_current"] is True

    resume_response = await client.get(f"/api/v1/resumes/{resume_id}", headers=headers)
    assert resume_response.json()["data"]["current_version"]["id"] == v1_id


async def test_delete_only_version_is_rejected(client: AsyncClient) -> None:
    headers = _user_header()
    upload = await _upload_resume(client, headers)
    resume_id = upload.json()["data"]["id"]
    version_id = upload.json()["data"]["current_version"]["id"]

    response = await client.delete(
        f"/api/v1/resumes/{resume_id}/versions/{version_id}", headers=headers
    )

    assert response.status_code == 409


async def test_delete_version_promotes_newest_remaining_to_current(client: AsyncClient) -> None:
    headers = _user_header()
    upload = await _upload_resume(client, headers)
    resume_id = upload.json()["data"]["id"]
    v1_id = upload.json()["data"]["current_version"]["id"]

    files = {"file": ("resume_v2.pdf", PDF_BYTES, "application/pdf")}
    v2 = await client.post(f"/api/v1/resumes/{resume_id}/versions", headers=headers, files=files)
    v2_id = v2.json()["data"]["id"]

    response = await client.delete(
        f"/api/v1/resumes/{resume_id}/versions/{v2_id}", headers=headers
    )
    assert response.status_code == 200

    resume_response = await client.get(f"/api/v1/resumes/{resume_id}", headers=headers)
    assert resume_response.json()["data"]["current_version"]["id"] == v1_id


async def test_delete_resume_then_get_returns_404(client: AsyncClient) -> None:
    headers = _user_header()
    upload = await _upload_resume(client, headers)
    resume_id = upload.json()["data"]["id"]

    delete_response = await client.delete(f"/api/v1/resumes/{resume_id}", headers=headers)
    assert delete_response.status_code == 200

    get_response = await client.get(f"/api/v1/resumes/{resume_id}", headers=headers)
    assert get_response.status_code == 404


async def test_download_url_returns_presigned_link(client: AsyncClient) -> None:
    headers = _user_header()
    upload = await _upload_resume(client, headers)
    resume_id = upload.json()["data"]["id"]
    version_id = upload.json()["data"]["current_version"]["id"]

    response = await client.get(
        f"/api/v1/resumes/{resume_id}/versions/{version_id}/download", headers=headers
    )

    assert response.status_code == 200
    assert response.json()["data"]["download_url"].startswith("https://")
