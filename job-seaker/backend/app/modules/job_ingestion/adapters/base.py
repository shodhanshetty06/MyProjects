"""ATS adapter base contract.

`BaseATSAdapter` is the Adapter Pattern seam: every concrete adapter takes
the same constructor shape and returns the same `JobPosting` DTO regardless
of how wildly different the underlying ATS API/HTML actually is. Everything
downstream (`tasks.py`, normalization, dedup) only ever depends on
`JobPosting`, never on a specific ATS's response shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel, Field

from app.modules.job_ingestion.models import ATSPlatform


class JobPosting(BaseModel):
    """Normalized-enough-to-store DTO returned by every adapter.

    Deliberately keeps raw/ambiguous fields as strings (`location_raw`,
    `employment_type_raw`, `salary_raw`) — turning those into structured
    data is `normalization.py`'s job, not the adapter's.
    """

    external_id: str
    title: str
    description_text: str
    description_html: str | None = None
    location_raw: str | None = None
    remote: bool | None = None
    employment_type_raw: str | None = None
    salary_raw: str | None = None
    posted_at: datetime | None = None
    url: str
    raw: dict[str, Any] = Field(default_factory=dict)


class BaseATSAdapter(ABC):
    """Common interface every concrete ATS adapter implements."""

    platform: ClassVar[ATSPlatform]

    def __init__(
        self,
        *,
        ats_identifier: str | None,
        career_url: str | None,
        adapter_config: dict[str, Any],
        http_client: httpx.AsyncClient,
    ) -> None:
        self.ats_identifier = ats_identifier
        self.career_url = career_url
        self.adapter_config = adapter_config
        self.http_client = http_client

    @abstractmethod
    async def fetch_jobs(self) -> list[JobPosting]:
        """Fetch and return every currently-open job posting for this company."""
        raise NotImplementedError
