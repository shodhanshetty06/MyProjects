"""Ashby Job Board API adapter.

Docs: https://developers.ashbyhq.com/reference/jobboardposting
No auth required for the public job board API — `ats_identifier` is the job
board name, i.e. the `X` in `https://jobs.ashbyhq.com/X`.
"""

from __future__ import annotations

from datetime import datetime

from bs4 import BeautifulSoup

from app.core.exceptions import ValidationError
from app.modules.job_ingestion.adapters.base import BaseATSAdapter, JobPosting
from app.modules.job_ingestion.models import ATSPlatform

_JOB_BOARD_URL = "https://api.ashbyhq.com/posting-api/job-board/{board_name}"


def _html_to_text(html: str | None) -> str:
    if not html:
        return ""
    return BeautifulSoup(html, "lxml").get_text(separator=" ", strip=True)


class AshbyAdapter(BaseATSAdapter):
    platform = ATSPlatform.ASHBY

    async def fetch_jobs(self) -> list[JobPosting]:
        if not self.ats_identifier:
            raise ValidationError(
                "Ashby adapter requires `ats_identifier` (the job board name).",
                error_code="missing_ats_identifier",
            )

        response = await self.http_client.get(
            _JOB_BOARD_URL.format(board_name=self.ats_identifier),
            params={"includeCompensation": "true"},
        )
        response.raise_for_status()
        payload = response.json()

        postings: list[JobPosting] = []
        for job in payload.get("jobs", []):
            posted_at = None
            if job.get("publishedAt"):
                posted_at = datetime.fromisoformat(job["publishedAt"].replace("Z", "+00:00"))

            location_parts = [job.get("location")]
            secondary = job.get("secondaryLocations") or []
            location_parts.extend(loc.get("location") for loc in secondary if loc.get("location"))
            location_raw = ", ".join(part for part in location_parts if part) or None

            postings.append(
                JobPosting(
                    external_id=str(job.get("id", "")),
                    title=job.get("title", ""),
                    description_text=_html_to_text(job.get("descriptionHtml")),
                    description_html=job.get("descriptionHtml"),
                    location_raw=location_raw,
                    remote=bool(job.get("isRemote")) or None,
                    employment_type_raw=job.get("employmentType"),
                    salary_raw=_format_compensation(job.get("compensation")),
                    posted_at=posted_at,
                    url=job.get("jobUrl", ""),
                    raw=job,
                )
            )
        return postings


def _format_compensation(compensation: dict | None) -> str | None:
    """Ashby's `compensation` block is a free-text summary plus a structured
    tiers list when `includeCompensation=true` is requested; the summary
    string is the more reliable field across boards."""
    if not compensation:
        return None
    summary = compensation.get("compensationTierSummary")
    if summary:
        return str(summary)
    tiers = compensation.get("summaryComponents") or []
    for tier in tiers:
        if tier.get("compensationTierSummary"):
            return str(tier["compensationTierSummary"])
    return None
