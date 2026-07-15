"""Lever Postings API adapter.

Docs: https://github.com/lever/postings-api
No auth required — `ats_identifier` is the company slug, i.e. the `X` in
`https://jobs.lever.co/X`.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.core.exceptions import ValidationError
from app.modules.job_ingestion.adapters.base import BaseATSAdapter, JobPosting
from app.modules.job_ingestion.models import ATSPlatform

_POSTINGS_URL = "https://api.lever.co/v0/postings/{company}"


class LeverAdapter(BaseATSAdapter):
    platform = ATSPlatform.LEVER

    async def fetch_jobs(self) -> list[JobPosting]:
        if not self.ats_identifier:
            raise ValidationError(
                "Lever adapter requires `ats_identifier` (the company slug).",
                error_code="missing_ats_identifier",
            )

        response = await self.http_client.get(
            _POSTINGS_URL.format(company=self.ats_identifier), params={"mode": "json"}
        )
        response.raise_for_status()
        postings_payload = response.json()

        postings: list[JobPosting] = []
        for posting in postings_payload:
            categories = posting.get("categories") or {}
            posted_at = None
            if posting.get("createdAt"):
                posted_at = datetime.fromtimestamp(posting["createdAt"] / 1000, tz=UTC)

            postings.append(
                JobPosting(
                    external_id=str(posting["id"]),
                    title=posting.get("text", ""),
                    description_text=posting.get("descriptionPlain", ""),
                    description_html=posting.get("description"),
                    location_raw=categories.get("location"),
                    remote=str(posting.get("workplaceType", "")).lower() == "remote" or None,
                    employment_type_raw=categories.get("commitment"),
                    salary_raw=_format_salary_range(posting.get("salaryRange")),
                    posted_at=posted_at,
                    url=posting.get("hostedUrl", ""),
                    raw=posting,
                )
            )
        return postings


def _format_salary_range(salary_range: dict | None) -> str | None:
    if not salary_range:
        return None
    minimum, maximum, currency = (
        salary_range.get("min"),
        salary_range.get("max"),
        salary_range.get("currency", ""),
    )
    if minimum is None and maximum is None:
        return None
    return f"{minimum}-{maximum} {currency}".strip()
