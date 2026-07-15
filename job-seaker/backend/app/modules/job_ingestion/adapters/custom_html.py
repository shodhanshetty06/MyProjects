"""Generic HTML scraping adapter — the fallback for companies with no known
ATS API.

Entirely driven by CSS selectors stored in `CompanyIntelligence.adapter_config`:

    {
        "job_list_selector": ".job-listing",       # required — one per posting
        "title_selector": ".job-title",             # required, relative to each listing
        "link_selector": "a",                       # required, relative to each listing (href)
        "location_selector": ".job-location",       # optional, relative to each listing
        "description_selector": ".job-description",  # optional, on the detail page
    }

Best-effort by nature: arbitrary company career pages have no stable
contract, so this adapter degrades gracefully (missing optional fields
become `None`) rather than failing the whole run over one bad listing.
"""

from __future__ import annotations

from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from app.core.exceptions import ValidationError
from app.modules.job_ingestion.adapters.base import BaseATSAdapter, JobPosting
from app.modules.job_ingestion.models import ATSPlatform

_REQUIRED_SELECTORS = ("job_list_selector", "title_selector", "link_selector")


class CustomHTMLAdapter(BaseATSAdapter):
    platform = ATSPlatform.CUSTOM_HTML

    async def fetch_jobs(self) -> list[JobPosting]:
        if not self.career_url:
            raise ValidationError(
                "Custom HTML adapter requires `career_url`.", error_code="missing_career_url"
            )
        missing = [key for key in _REQUIRED_SELECTORS if not self.adapter_config.get(key)]
        if missing:
            missing_keys = ", ".join(missing)
            raise ValidationError(
                f"Custom HTML adapter is missing required adapter_config keys: {missing_keys}.",
                error_code="missing_adapter_config",
            )

        response = await self.http_client.get(self.career_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        listings = soup.select(self.adapter_config["job_list_selector"])
        postings: list[JobPosting] = []
        for listing in listings:
            posting = await self._parse_listing(listing)
            if posting is not None:
                postings.append(posting)
        return postings

    async def _parse_listing(self, listing: Tag) -> JobPosting | None:
        title_el = listing.select_one(self.adapter_config["title_selector"])
        link_el = listing.select_one(self.adapter_config["link_selector"])
        if title_el is None or link_el is None or not link_el.get("href"):
            return None

        title = title_el.get_text(strip=True)
        url = urljoin(self.career_url or "", str(link_el["href"]))

        location_raw = None
        location_selector = self.adapter_config.get("location_selector")
        if location_selector:
            location_el = listing.select_one(location_selector)
            location_raw = location_el.get_text(strip=True) if location_el else None

        description_text, description_html = await self._fetch_description(url)

        return JobPosting(
            external_id=url,
            title=title,
            description_text=description_text,
            description_html=description_html,
            location_raw=location_raw,
            url=url,
            raw={"listing_html": str(listing)},
        )

    async def _fetch_description(self, detail_url: str) -> tuple[str, str | None]:
        description_selector = self.adapter_config.get("description_selector")
        if not description_selector:
            return "", None

        response = await self.http_client.get(detail_url)
        if response.status_code != 200:
            return "", None

        detail_soup = BeautifulSoup(response.text, "lxml")
        description_el = detail_soup.select_one(description_selector)
        if description_el is None:
            return "", None
        return description_el.get_text(separator=" ", strip=True), str(description_el)
