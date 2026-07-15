"""Apply adapter package — the Adapter Pattern implementation for Auto Apply.

Mirrors `job_ingestion.adapters`: each supported ATS gets one
`BaseApplyAdapter` subclass, and `get_apply_adapter()` is the single factory
entry point `tasks.py` uses — callers never import a concrete adapter class
directly. `WORKDAY` and any platform without a known API
(`CUSTOM_HTML`/`UNKNOWN`) both drive a real headless browser (see
`browser.py`); `GREENHOUSE`/`LEVER` submit plain HTML forms over HTTP.
"""
from __future__ import annotations

import httpx

from app.core.exceptions import ValidationError
from app.modules.auto_apply.adapters.base import ApplicationField, ApplyResult, BaseApplyAdapter, ResumeAttachment
from app.modules.auto_apply.adapters.greenhouse import GreenhouseApplyAdapter
from app.modules.auto_apply.adapters.lever import LeverApplyAdapter
from app.modules.auto_apply.adapters.playwright_generic import PlaywrightApplyAdapter
from app.modules.auto_apply.adapters.workday import WorkdayApplyAdapter
from app.modules.job_ingestion.models import ATSPlatform

_HTTP_ADAPTERS: dict[ATSPlatform, type[BaseApplyAdapter]] = {
    ATSPlatform.GREENHOUSE: GreenhouseApplyAdapter,
    ATSPlatform.LEVER: LeverApplyAdapter,
}
_BROWSER_ADAPTERS: dict[ATSPlatform, type[BaseApplyAdapter]] = {
    ATSPlatform.WORKDAY: WorkdayApplyAdapter,
}
_FALLBACK_ADAPTER: type[BaseApplyAdapter] = PlaywrightApplyAdapter


def get_apply_adapter(
    platform: ATSPlatform,
    *,
    job_url: str,
    external_id: str,
    ats_identifier: str | None,
    adapter_config: dict,
    http_client: httpx.AsyncClient | None = None,
) -> BaseApplyAdapter:
    """Instantiate the adapter registered for `platform`.

    `CUSTOM_HTML` and `UNKNOWN` both fall through to the generic
    Playwright-driven adapter — for `UNKNOWN` specifically this is a
    best-effort attempt (no ATS API/known selectors), so `field_mapping.py`
    downstream will typically surface most fields as missing unless
    `adapter_config.apply` was hand-configured for that company.
    """
    adapter_cls = _HTTP_ADAPTERS.get(platform) or _BROWSER_ADAPTERS.get(platform) or _FALLBACK_ADAPTER

    if adapter_cls in _HTTP_ADAPTERS.values() and http_client is None:
        raise ValidationError(
            f"'{platform.value}' apply adapter requires an http_client.",
            error_code="missing_http_client",
        )

    return adapter_cls(
        job_url=job_url,
        external_id=external_id,
        ats_identifier=ats_identifier,
        adapter_config=adapter_config,
        http_client=http_client,
    )


__all__ = ["ApplicationField", "ApplyResult", "BaseApplyAdapter", "ResumeAttachment", "get_apply_adapter"]
