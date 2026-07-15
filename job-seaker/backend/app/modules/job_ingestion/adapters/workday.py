"""Workday CXS (careers hub) API adapter.

Workday has no single public API shape — every tenant runs its own CXS
endpoint at `https://{tenant}.wd1.myworkdayjobs.com/wday/cxs/{tenant}/{site}`.
`ats_identifier` therefore encodes both parts as `"{tenant}/{site}"`.

The search endpoint returns only a summary per posting (title, relative
"posted X days ago" text, no description), so this adapter follows up with
one detail request per posting to get the real description/employment type.
Total requests are bounded by `settings.JOB_INGESTION_MAX_JOBS_PER_RUN`.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

from bs4 import BeautifulSoup

from app.core.config import get_settings
from app.core.exceptions import ValidationError
from app.modules.job_ingestion.adapters.base import BaseATSAdapter, JobPosting
from app.modules.job_ingestion.models import ATSPlatform

_PAGE_SIZE = 20
_RELATIVE_DAYS_RE = re.compile(r"posted\s+(\d+)\s+day", re.IGNORECASE)


def _base_url(tenant: str) -> str:
    return f"https://{tenant}.wd1.myworkdayjobs.com"


def _parse_relative_posted(text: str | None) -> datetime | None:
    if not text:
        return None
    text_lower = text.lower()
    now = datetime.now(UTC)
    if "today" in text_lower:
        return now
    match = _RELATIVE_DAYS_RE.search(text_lower)
    if match:
        return now - timedelta(days=int(match.group(1)))
    return None


def _html_to_text(html: str | None) -> str:
    if not html:
        return ""
    return BeautifulSoup(html, "lxml").get_text(separator=" ", strip=True)


class WorkdayAdapter(BaseATSAdapter):
    platform = ATSPlatform.WORKDAY

    async def fetch_jobs(self) -> list[JobPosting]:
        if not self.ats_identifier or "/" not in self.ats_identifier:
            raise ValidationError(
                "Workday adapter requires `ats_identifier` in the form '{tenant}/{site}'.",
                error_code="missing_ats_identifier",
            )
        tenant, site = self.ats_identifier.split("/", 1)
        settings = get_settings()
        base = _base_url(tenant)
        cxs_url = f"{base}/wday/cxs/{tenant}/{site}/jobs"

        summaries: list[dict] = []
        offset = 0
        total = None
        while total is None or (
            offset < total and len(summaries) < settings.JOB_INGESTION_MAX_JOBS_PER_RUN
        ):
            response = await self.http_client.post(
                cxs_url,
                json={"appliedFacets": {}, "limit": _PAGE_SIZE, "offset": offset, "searchText": ""},
            )
            response.raise_for_status()
            payload = response.json()
            total = payload.get("total", 0)
            page = payload.get("jobPostings", [])
            if not page:
                break
            summaries.extend(page)
            offset += _PAGE_SIZE

        postings: list[JobPosting] = []
        for summary in summaries[: settings.JOB_INGESTION_MAX_JOBS_PER_RUN]:
            external_path = summary.get("externalPath", "")
            detail = await self._fetch_detail(base, tenant, site, external_path)
            job_info = (detail or {}).get("jobPostingInfo", {})

            bullet_fields = summary.get("bulletFields") or []
            external_id = bullet_fields[0] if bullet_fields else external_path

            postings.append(
                JobPosting(
                    external_id=str(external_id),
                    title=summary.get("title", ""),
                    description_text=_html_to_text(job_info.get("jobDescription")),
                    description_html=job_info.get("jobDescription"),
                    location_raw=summary.get("locationsText") or job_info.get("location"),
                    employment_type_raw=job_info.get("timeType"),
                    salary_raw=None,
                    posted_at=_parse_relative_posted(summary.get("postedOn")),
                    url=f"{base}/{site}{external_path}",
                    raw={"summary": summary, "detail": detail},
                )
            )
        return postings

    async def _fetch_detail(
        self, base: str, tenant: str, site: str, external_path: str
    ) -> dict | None:
        if not external_path:
            return None
        detail_url = f"{base}/wday/cxs/{tenant}/{site}{external_path}"
        response = await self.http_client.get(detail_url)
        if response.status_code != 200:
            return None
        return response.json()
