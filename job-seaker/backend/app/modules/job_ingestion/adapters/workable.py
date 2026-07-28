"""Workable public job board widget API adapter.

Docs (unofficial but stable/widely used): the public "widget" JSON endpoint
backing every `apply.workable.com/{account}` board. `ats_identifier` is the
account subdomain, i.e. the `X` in `https://apply.workable.com/X/`.

The listing endpoint returns summaries only; each job's full description
requires a follow-up per-job request, mirroring the Workday adapter's
two-step shape. Requests are capped by
`settings.JOB_INGESTION_MAX_JOBS_PER_RUN`.
"""

from __future__ import annotations

from datetime import UTC, datetime

from bs4 import BeautifulSoup

from app.core.config import get_settings
from app.core.exceptions import ValidationError
from app.modules.job_ingestion.adapters.base import BaseATSAdapter, JobPosting
from app.modules.job_ingestion.models import ATSPlatform

_LISTING_URL = "https://www.workable.com/api/v1/widget/accounts/{account}"
_DETAIL_URL = "https://www.workable.com/api/v1/widget/accounts/{account}/jobs/{shortcode}"


def _html_to_text(html: str | None) -> str:
    if not html:
        return ""
    return BeautifulSoup(html, "lxml").get_text(separator=" ", strip=True)


def _location_raw(job: dict) -> str | None:
    parts = [job.get("city"), job.get("region"), job.get("country")]
    text = ", ".join(part for part in parts if part)
    return text or None


class WorkableAdapter(BaseATSAdapter):
    platform = ATSPlatform.WORKABLE

    async def fetch_jobs(self) -> list[JobPosting]:
        if not self.ats_identifier:
            raise ValidationError(
                "Workable adapter requires `ats_identifier` (the account subdomain).",
                error_code="missing_ats_identifier",
            )
        settings = get_settings()

        response = await self.http_client.get(
            _LISTING_URL.format(account=self.ats_identifier), params={"details": "true"}
        )
        response.raise_for_status()
        payload = response.json()
        jobs = (payload.get("jobs") or [])[: settings.JOB_INGESTION_MAX_JOBS_PER_RUN]

        postings: list[JobPosting] = []
        for job in jobs:
            shortcode = job.get("shortcode")
            detail = await self._fetch_detail(shortcode) if shortcode else None
            description_html = (detail or {}).get("description") or job.get("description")

            posted_at = None
            if job.get("published_on"):
                posted_at = datetime.fromisoformat(job["published_on"]).replace(tzinfo=UTC)

            postings.append(
                JobPosting(
                    external_id=str(shortcode or job.get("id", "")),
                    title=job.get("title", ""),
                    description_text=_html_to_text(description_html),
                    description_html=description_html,
                    location_raw=_location_raw(job),
                    remote=bool(job.get("telecommute")) or None,
                    employment_type_raw=job.get("employment_type"),
                    salary_raw=None,
                    posted_at=posted_at,
                    url=job.get("url", ""),
                    raw={"listing": job, "detail": detail},
                )
            )
        return postings

    async def _fetch_detail(self, shortcode: str) -> dict | None:
        response = await self.http_client.get(
            _DETAIL_URL.format(account=self.ats_identifier, shortcode=shortcode)
        )
        if response.status_code != 200:
            return None
        return response.json()
