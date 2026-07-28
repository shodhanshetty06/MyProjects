"""schema.org `JobPosting` JSON-LD adapter.

Jobvite (and most other ATS-hosted career pages, for SEO/Google-for-Jobs
purposes) embed a `<script type="application/ld+json">` block per job detail
page following the https://schema.org/JobPosting vocabulary. Unlike
Greenhouse/Lever/Ashby/Workable/SmartRecruiters, Jobvite has no single stable,
publicly documented REST API across tenants — rather than guess at one and
ship something that silently breaks, this adapter reads the same structured
data Google indexes, which is a real, standardized contract every one of
these pages already commits to maintaining.

Two-step: discover job detail URLs from the listing page (via CSS selectors,
same shape as `custom_html.py`), then parse the JSON-LD out of each detail
page. `adapter_config`:

    {
        "job_list_selector": ".jv-job-list-name",   # required — one per posting
        "link_selector": "a",                        # required, relative to each listing (href)
    }

Falls back gracefully per-listing (skips postings with no parseable
JobPosting block) rather than failing the whole run.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from app.core.exceptions import ValidationError
from app.modules.job_ingestion.adapters.base import BaseATSAdapter, JobPosting
from app.modules.job_ingestion.models import ATSPlatform

_REQUIRED_SELECTORS = ("job_list_selector", "link_selector")


def _first(value: Any) -> Any:
    """schema.org allows single objects or arrays interchangeably."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _find_job_posting_nodes(payload: Any) -> list[dict]:
    """Walk a parsed JSON-LD payload (object, array, or `@graph` wrapper) and
    return every node whose `@type` is (or includes) `JobPosting`."""
    nodes: list[dict] = []

    def _walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                _walk(item)
            return
        if not isinstance(node, dict):
            return
        node_type = node.get("@type")
        types = node_type if isinstance(node_type, list) else [node_type]
        if "JobPosting" in types:
            nodes.append(node)
        if "@graph" in node:
            _walk(node["@graph"])

    _walk(payload)
    return nodes


def _location_raw(job_location: Any) -> str | None:
    place = _first(job_location)
    if not isinstance(place, dict):
        return None
    address = place.get("address")
    if isinstance(address, dict):
        parts = [
            address.get("addressLocality"),
            address.get("addressRegion"),
            address.get("addressCountry"),
        ]
        text = ", ".join(str(p) for p in parts if p)
        return text or None
    if isinstance(address, str):
        return address
    return None


def _salary_raw(base_salary: Any) -> str | None:
    if not isinstance(base_salary, dict):
        return None
    value = base_salary.get("value")
    currency = base_salary.get("currency", "")
    if not isinstance(value, dict):
        return None
    minimum, maximum = value.get("minValue"), value.get("maxValue")
    unit = value.get("unitText", "")
    if minimum is None and maximum is None:
        single = value.get("value")
        if single is None:
            return None
        return f"{single} {currency} {unit}".strip()
    return f"{minimum}-{maximum} {currency} {unit}".strip()


def _parse_job_posting_node(node: dict, *, fallback_url: str) -> JobPosting | None:
    title = node.get("title")
    description = node.get("description")
    if not title or not description:
        return None

    posted_at = None
    if node.get("datePosted"):
        try:
            posted_at = datetime.fromisoformat(str(node["datePosted"]).replace("Z", "+00:00"))
        except ValueError:
            posted_at = None

    is_remote = str(node.get("jobLocationType", "")).upper() == "TELECOMMUTE"
    identifier = node.get("identifier")
    external_id = fallback_url
    if isinstance(identifier, dict):
        external_id = str(identifier.get("value") or fallback_url)
    elif isinstance(identifier, str):
        external_id = identifier

    description_text = BeautifulSoup(str(description), "lxml").get_text(
        separator=" ", strip=True
    )

    return JobPosting(
        external_id=external_id,
        title=str(title),
        description_text=description_text,
        description_html=str(description),
        location_raw=_location_raw(node.get("jobLocation")),
        remote=is_remote or None,
        employment_type_raw=str(node.get("employmentType") or "") or None,
        salary_raw=_salary_raw(node.get("baseSalary")),
        posted_at=posted_at,
        url=str(node.get("url") or fallback_url),
        raw=node,
    )


class JsonLdJobPostingAdapter(BaseATSAdapter):
    """Base class parameterized by `platform`; `JobviteAdapter` below is the
    concrete registration but this also serves as the fallback for any custom
    career page that emits schema.org JobPosting markup."""

    platform = ATSPlatform.JOBVITE

    async def fetch_jobs(self) -> list[JobPosting]:
        if not self.career_url:
            raise ValidationError(
                "JSON-LD adapter requires `career_url`.", error_code="missing_career_url"
            )
        missing = [key for key in _REQUIRED_SELECTORS if not self.adapter_config.get(key)]
        if missing:
            missing_keys = ", ".join(missing)
            raise ValidationError(
                f"JSON-LD adapter is missing required adapter_config keys: {missing_keys}.",
                error_code="missing_adapter_config",
            )

        response = await self.http_client.get(self.career_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        listings = soup.select(self.adapter_config["job_list_selector"])
        job_urls: list[str] = []
        for listing in listings:
            link_el = listing.select_one(self.adapter_config["link_selector"])
            if isinstance(link_el, Tag) and link_el.get("href"):
                job_urls.append(urljoin(self.career_url, str(link_el["href"])))

        postings: list[JobPosting] = []
        for url in job_urls:
            posting = await self._fetch_posting(url)
            if posting is not None:
                postings.append(posting)
        return postings

    async def _fetch_posting(self, url: str) -> JobPosting | None:
        response = await self.http_client.get(url)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.text, "lxml")

        for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
            try:
                payload = json.loads(script.string or "")
            except (json.JSONDecodeError, TypeError):
                continue
            for node in _find_job_posting_nodes(payload):
                posting = _parse_job_posting_node(node, fallback_url=url)
                if posting is not None:
                    return posting
        return None


class JobviteAdapter(JsonLdJobPostingAdapter):
    platform = ATSPlatform.JOBVITE
