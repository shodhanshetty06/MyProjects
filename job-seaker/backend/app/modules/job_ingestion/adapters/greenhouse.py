"""Greenhouse Job Board API adapter.

Docs: https://developers.greenhouse.io/job-board.html
No auth required for the public job board API — `ats_identifier` is the
board token, i.e. the `X` in `https://boards.greenhouse.io/X`.
"""

from __future__ import annotations

from datetime import datetime

from bs4 import BeautifulSoup

from app.core.exceptions import ValidationError
from app.modules.job_ingestion.adapters.base import BaseATSAdapter, JobPosting
from app.modules.job_ingestion.models import ATSPlatform

_BOARD_URL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs"


def _html_to_text(html: str | None) -> str:
    if not html:
        return ""
    return BeautifulSoup(html, "lxml").get_text(separator=" ", strip=True)


class GreenhouseAdapter(BaseATSAdapter):
    platform = ATSPlatform.GREENHOUSE

    async def fetch_jobs(self) -> list[JobPosting]:
        if not self.ats_identifier:
            raise ValidationError(
                "Greenhouse adapter requires `ats_identifier` (the board token).",
                error_code="missing_ats_identifier",
            )

        response = await self.http_client.get(
            _BOARD_URL.format(token=self.ats_identifier), params={"content": "true"}
        )
        response.raise_for_status()
        payload = response.json()

        postings: list[JobPosting] = []
        for job in payload.get("jobs", []):
            location = (job.get("location") or {}).get("name")
            posted_at = None
            if job.get("updated_at"):
                posted_at = datetime.fromisoformat(job["updated_at"].replace("Z", "+00:00"))

            postings.append(
                JobPosting(
                    external_id=str(job["id"]),
                    title=job.get("title", ""),
                    description_text=_html_to_text(job.get("content")),
                    description_html=job.get("content"),
                    location_raw=location,
                    employment_type_raw=None,
                    salary_raw=_extract_salary_from_metadata(job.get("metadata")),
                    posted_at=posted_at,
                    url=job.get("absolute_url", ""),
                    raw=job,
                )
            )
        return postings


def _extract_salary_from_metadata(metadata: list[dict] | None) -> str | None:
    """Greenhouse has no dedicated salary field — some boards expose it as a
    custom `metadata` question (e.g. name "Salary Range")."""
    if not metadata:
        return None
    for field in metadata:
        name = str(field.get("name", "")).lower()
        if "salary" in name or "compensation" in name:
            value = field.get("value")
            return str(value) if value else None
    return None
