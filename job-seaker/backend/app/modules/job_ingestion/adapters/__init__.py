"""ATS adapter package — the Adapter Pattern implementation for Job Ingestion.

Each supported ATS gets one `BaseATSAdapter` subclass that knows how to turn
that platform's job board into a list of normalized `JobPosting` DTOs.
`get_adapter()` is the single factory entry point `tasks.py` uses; callers
never import a concrete adapter class directly.
"""

from __future__ import annotations

import httpx

from app.core.exceptions import ValidationError
from app.modules.job_ingestion.adapters.base import BaseATSAdapter, JobPosting
from app.modules.job_ingestion.adapters.custom_html import CustomHTMLAdapter
from app.modules.job_ingestion.adapters.greenhouse import GreenhouseAdapter
from app.modules.job_ingestion.adapters.lever import LeverAdapter
from app.modules.job_ingestion.adapters.workday import WorkdayAdapter
from app.modules.job_ingestion.models import ATSPlatform

_ADAPTERS: dict[ATSPlatform, type[BaseATSAdapter]] = {
    ATSPlatform.GREENHOUSE: GreenhouseAdapter,
    ATSPlatform.LEVER: LeverAdapter,
    ATSPlatform.WORKDAY: WorkdayAdapter,
    ATSPlatform.CUSTOM_HTML: CustomHTMLAdapter,
}


def get_adapter(
    platform: ATSPlatform,
    *,
    ats_identifier: str | None,
    career_url: str | None,
    adapter_config: dict,
    http_client: httpx.AsyncClient,
) -> BaseATSAdapter:
    """Instantiate the adapter registered for `platform`.

    Raises `ValidationError` for `UNKNOWN` or any platform without a
    registered adapter, so callers get a clear 4xx-mappable error instead of
    a `KeyError`.
    """
    adapter_cls = _ADAPTERS.get(platform)
    if adapter_cls is None:
        raise ValidationError(
            f"No ATS adapter is registered for platform '{platform.value}'.",
            error_code="unsupported_ats_platform",
        )
    return adapter_cls(
        ats_identifier=ats_identifier,
        career_url=career_url,
        adapter_config=adapter_config,
        http_client=http_client,
    )


__all__ = ["BaseATSAdapter", "JobPosting", "get_adapter"]
