"""SmartRecruiters Posting API adapter.

Docs: https://developers.smartrecruiters.com/docs/job-postings-api
No auth required for public postings — `ats_identifier` is the company
identifier, i.e. the `X` in `https://api.smartrecruiters.com/v1/companies/X`.

The list endpoint returns summaries only; each posting's full `jobAd`
(description/qualifications/responsibilities sections) requires a follow-up
per-posting detail request. Requests are capped by
`settings.JOB_INGESTION_MAX_JOBS_PER_RUN`.
"""

from __future__ import annotations

from datetime import datetime

from app.core.config import get_settings
from app.core.exceptions import ValidationError
from app.modules.job_ingestion.adapters.base import BaseATSAdapter, JobPosting
from app.modules.job_ingestion.models import ATSPlatform

_LIST_URL = "https://api.smartrecruiters.com/v1/companies/{company}/postings"
_DETAIL_URL = "https://api.smartrecruiters.com/v1/companies/{company}/postings/{posting_id}"
_PAGE_LIMIT = 100


def _location_raw(location: dict | None) -> str | None:
    if not location:
        return None
    parts = [location.get("city"), location.get("region"), location.get("country")]
    text = ", ".join(part for part in parts if part)
    return text or None


def _description_text(job_ad: dict | None) -> str:
    if not job_ad:
        return ""
    sections = (job_ad.get("sections") or {}).values()
    chunks: list[str] = []
    for section in sections:
        text = section.get("text")
        if text:
            chunks.append(text)
    return "\n\n".join(chunks)


class SmartRecruitersAdapter(BaseATSAdapter):
    platform = ATSPlatform.SMARTRECRUITERS

    async def fetch_jobs(self) -> list[JobPosting]:
        if not self.ats_identifier:
            raise ValidationError(
                "SmartRecruiters adapter requires `ats_identifier` (the company identifier).",
                error_code="missing_ats_identifier",
            )
        settings = get_settings()

        summaries: list[dict] = []
        offset = 0
        while len(summaries) < settings.JOB_INGESTION_MAX_JOBS_PER_RUN:
            response = await self.http_client.get(
                _LIST_URL.format(company=self.ats_identifier),
                params={"limit": _PAGE_LIMIT, "offset": offset},
            )
            response.raise_for_status()
            payload = response.json()
            page = payload.get("content", [])
            if not page:
                break
            summaries.extend(page)
            offset += _PAGE_LIMIT
            if offset >= payload.get("totalFound", 0):
                break

        postings: list[JobPosting] = []
        for summary in summaries[: settings.JOB_INGESTION_MAX_JOBS_PER_RUN]:
            posting_id = str(summary.get("id", ""))
            detail = await self._fetch_detail(posting_id) if posting_id else None
            job_ad = (detail or {}).get("jobAd", {}).get("sections") and (detail or {}).get(
                "jobAd"
            )

            posted_at = None
            if summary.get("releasedDate"):
                posted_at = datetime.fromisoformat(
                    summary["releasedDate"].replace("Z", "+00:00")
                )

            employment_type = (summary.get("typeOfEmployment") or {}).get("label")
            location = summary.get("location") or {}

            postings.append(
                JobPosting(
                    external_id=posting_id,
                    title=summary.get("name", ""),
                    description_text=_description_text(job_ad),
                    description_html=None,
                    location_raw=_location_raw(location),
                    remote=bool(location.get("remote")) or None,
                    employment_type_raw=employment_type,
                    salary_raw=None,
                    posted_at=posted_at,
                    url=(summary.get("ref") or (detail or {}).get("ref") or ""),
                    raw={"summary": summary, "detail": detail},
                )
            )
        return postings

    async def _fetch_detail(self, posting_id: str) -> dict | None:
        response = await self.http_client.get(
            _DETAIL_URL.format(company=self.ats_identifier, posting_id=posting_id)
        )
        if response.status_code != 200:
            return None
        return response.json()
