"""Unit tests for the HTTP-based apply adapters (Greenhouse, Lever) — each
fed a mocked `httpx` transport so no real network access is required, same
convention as `tests/modules/job_ingestion/test_adapters.py`. Workday and
the generic Playwright adapter drive a real headless browser and aren't
covered here (no browser available in this test environment).
"""
from __future__ import annotations

import httpx
import pytest

from app.modules.auto_apply.adapters.base import ResumeAttachment
from app.modules.auto_apply.adapters.greenhouse import GreenhouseApplyAdapter
from app.modules.auto_apply.adapters.html_form import build_multipart_data, find_form, parse_form_fields
from app.modules.auto_apply.adapters.lever import LeverApplyAdapter

_GREENHOUSE_APPLY_HTML = """
<html><body>
  <form id="application-form" action="/acme/jobs/123" method="post">
    <label for="first_name">First Name*</label>
    <input type="text" id="first_name" name="first_name" required />
    <label for="email">Email</label>
    <input type="email" id="email" name="email" required />
    <input type="file" name="resume" />
  </form>
</body></html>
"""

_LEVER_APPLY_HTML = """
<html><body>
  <form class="application-form" action="/acme/abc/apply" method="post">
    <label for="name">Full Name</label>
    <input type="text" id="name" name="name" required />
    <input type="file" name="resume" />
  </form>
</body></html>
"""


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://boards.greenhouse.io")


def _resume() -> ResumeAttachment:
    return ResumeAttachment(filename="resume.pdf", content_type="application/pdf", content=b"%PDF-1.4")


def test_parse_form_fields_infers_types_labels_and_required() -> None:
    form = find_form(_GREENHOUSE_APPLY_HTML)
    assert form is not None
    fields = parse_form_fields(form)

    by_key = {f.key: f for f in fields}
    assert by_key["first_name"].label == "First Name*"
    assert by_key["first_name"].required is True
    assert by_key["email"].field_type.value == "email"
    assert by_key["resume"].field_type.value == "file"


def test_build_multipart_data_splits_resume_from_fields() -> None:
    data, files = build_multipart_data(
        field_values={"first_name": "Ada", "email": "ada@example.com"}, resume=_resume()
    )
    assert data == {"first_name": "Ada", "email": "ada@example.com"}
    assert files["resume"][0] == "resume.pdf"


@pytest.mark.asyncio
async def test_greenhouse_adapter_discovers_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_GREENHOUSE_APPLY_HTML)

    adapter = GreenhouseApplyAdapter(
        job_url="https://boards.greenhouse.io/acme/jobs/123",
        external_id="123",
        ats_identifier="acme",
        adapter_config={},
        http_client=_client(handler),
    )
    fields = await adapter.discover_fields()
    assert {f.key for f in fields} == {"first_name", "email", "resume"}


@pytest.mark.asyncio
async def test_greenhouse_adapter_submits_and_detects_confirmation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, text=_GREENHOUSE_APPLY_HTML)
        assert request.url.path == "/acme/jobs/123"
        return httpx.Response(200, text="<html><body>Thank you for applying!</body></html>")

    adapter = GreenhouseApplyAdapter(
        job_url="https://boards.greenhouse.io/acme/jobs/123",
        external_id="123",
        ats_identifier="acme",
        adapter_config={},
        http_client=_client(handler),
    )
    result = await adapter.apply(
        field_values={"first_name": "Ada", "email": "ada@example.com"}, resume=_resume()
    )
    assert result.success is True
    assert result.confirmation_detected is True


@pytest.mark.asyncio
async def test_lever_adapter_discovers_fields_and_applies() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, text=_LEVER_APPLY_HTML)
        return httpx.Response(200, text="<html><body>Application received</body></html>")

    adapter = LeverApplyAdapter(
        job_url="https://jobs.lever.co/acme/abc",
        external_id="abc",
        ats_identifier="acme",
        adapter_config={},
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://jobs.lever.co"
        ),
    )
    fields = await adapter.discover_fields()
    assert {f.key for f in fields} == {"name", "resume"}

    result = await adapter.apply(field_values={"name": "Ada Lovelace"}, resume=_resume())
    assert result.success is True
    assert result.confirmation_detected is True


@pytest.mark.asyncio
async def test_greenhouse_adapter_reports_failure_on_bad_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, text=_GREENHOUSE_APPLY_HTML)
        return httpx.Response(422, text="<html><body>Please fix errors</body></html>")

    adapter = GreenhouseApplyAdapter(
        job_url="https://boards.greenhouse.io/acme/jobs/123",
        external_id="123",
        ats_identifier="acme",
        adapter_config={},
        http_client=_client(handler),
    )
    result = await adapter.apply(field_values={"first_name": "Ada"}, resume=_resume())
    assert result.success is False
    assert result.confirmation_detected is False
