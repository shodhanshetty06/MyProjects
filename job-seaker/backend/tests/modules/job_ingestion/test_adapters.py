"""Unit tests for ATS adapters — each fed a mocked `httpx` transport so no
real network access is required.
"""

from __future__ import annotations

import httpx
import pytest

from app.modules.job_ingestion.adapters.custom_html import CustomHTMLAdapter
from app.modules.job_ingestion.adapters.greenhouse import GreenhouseAdapter
from app.modules.job_ingestion.adapters.lever import LeverAdapter


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_greenhouse_adapter_maps_jobs() -> None:
    payload = {
        "jobs": [
            {
                "id": 123,
                "title": "Software Engineer",
                "absolute_url": "https://boards.greenhouse.io/acme/jobs/123",
                "location": {"name": "Remote"},
                "updated_at": "2026-07-01T12:00:00Z",
                "content": "<p>Great job</p>",
                "metadata": [{"name": "Salary Range", "value": "$100k-$120k"}],
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/boards/acme/jobs"
        return httpx.Response(200, json=payload)

    adapter = GreenhouseAdapter(
        ats_identifier="acme", career_url=None, adapter_config={}, http_client=_client(handler)
    )
    postings = await adapter.fetch_jobs()

    assert len(postings) == 1
    posting = postings[0]
    assert posting.external_id == "123"
    assert posting.title == "Software Engineer"
    assert posting.description_text == "Great job"
    assert posting.location_raw == "Remote"
    assert posting.salary_raw == "$100k-$120k"
    assert posting.url == "https://boards.greenhouse.io/acme/jobs/123"


@pytest.mark.asyncio
async def test_lever_adapter_maps_postings() -> None:
    payload = [
        {
            "id": "abc",
            "text": "Data Scientist",
            "hostedUrl": "https://jobs.lever.co/acme/abc",
            "categories": {"location": "NYC", "commitment": "Full-time"},
            "descriptionPlain": "Do data science",
            "description": "<p>Do data science</p>",
            "createdAt": 1700000000000,
            "workplaceType": "remote",
            "salaryRange": {"min": 100000, "max": 150000, "currency": "USD"},
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v0/postings/acme"
        return httpx.Response(200, json=payload)

    adapter = LeverAdapter(
        ats_identifier="acme", career_url=None, adapter_config={}, http_client=_client(handler)
    )
    postings = await adapter.fetch_jobs()

    assert len(postings) == 1
    posting = postings[0]
    assert posting.external_id == "abc"
    assert posting.title == "Data Scientist"
    assert posting.description_text == "Do data science"
    assert posting.location_raw == "NYC"
    assert posting.employment_type_raw == "Full-time"
    assert posting.remote is True
    assert posting.salary_raw == "100000-150000 USD"


@pytest.mark.asyncio
async def test_custom_html_adapter_scrapes_listings() -> None:
    listing_html = """
    <html><body>
      <div class="job-listing">
        <a class="job-title" href="/jobs/1">Backend Engineer</a>
        <span class="job-location">Berlin, Germany</span>
      </div>
    </body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=listing_html)

    adapter = CustomHTMLAdapter(
        ats_identifier=None,
        career_url="https://example.com/careers",
        adapter_config={
            "job_list_selector": ".job-listing",
            "title_selector": ".job-title",
            "link_selector": ".job-title",
            "location_selector": ".job-location",
        },
        http_client=_client(handler),
    )
    postings = await adapter.fetch_jobs()

    assert len(postings) == 1
    posting = postings[0]
    assert posting.title == "Backend Engineer"
    assert posting.url == "https://example.com/jobs/1"
    assert posting.location_raw == "Berlin, Germany"


def test_lever_format_salary_range_handles_missing_values() -> None:
    from app.modules.job_ingestion.adapters.lever import _format_salary_range

    assert _format_salary_range(None) is None
    assert _format_salary_range({"min": None, "max": None, "currency": "USD"}) is None
    assert _format_salary_range({"min": 1, "max": 2, "currency": "USD"}) == "1-2 USD"


def test_greenhouse_extract_salary_from_metadata() -> None:
    from app.modules.job_ingestion.adapters.greenhouse import _extract_salary_from_metadata

    assert _extract_salary_from_metadata(None) is None
    assert _extract_salary_from_metadata([{"name": "Team", "value": "Eng"}]) is None
    assert _extract_salary_from_metadata([{"name": "Compensation", "value": "$100k"}]) == "$100k"
