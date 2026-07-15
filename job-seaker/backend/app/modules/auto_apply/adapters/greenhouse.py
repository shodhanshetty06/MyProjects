"""Greenhouse apply adapter.

Greenhouse's hosted apply page (the job's `source_url` from
`job_ingestion.GreenhouseAdapter`) is a plain server-rendered `<form>` — no
SPA framework needed to fill and submit it — so this adapter talks HTTP
directly via `httpx`, the same style as `job_ingestion.adapters.greenhouse`,
rather than driving a real browser.
"""
from __future__ import annotations

from urllib.parse import urljoin

from app.core.exceptions import ServiceUnavailableError
from app.modules.auto_apply.adapters.base import (
    ApplicationField,
    ApplyLogEntry,
    ApplyResult,
    BaseApplyAdapter,
    ResumeAttachment,
)
from app.modules.auto_apply.adapters.html_form import build_multipart_data, find_form, parse_form_fields
from app.modules.auto_apply.confirmation import detect_confirmation
from app.modules.job_ingestion.models import ATSPlatform

_RESUME_FIELD_CANDIDATES = ("resume", "job_application[resume]")


class GreenhouseApplyAdapter(BaseApplyAdapter):
    platform = ATSPlatform.GREENHOUSE

    async def _fetch_form(self):
        assert self.http_client is not None
        try:
            response = await self.http_client.get(self.job_url)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 -- reclassified by errors.classify_exception upstream
            raise ServiceUnavailableError(
                f"Could not load the Greenhouse apply page: {exc}"
            ) from exc

        form = find_form(response.text, form_selector="#application-form, form#application_form, form")
        if form is None:
            raise ServiceUnavailableError("Greenhouse apply page has no recognizable application form.")
        return response, form

    async def discover_fields(self) -> list[ApplicationField]:
        _, form = await self._fetch_form()
        return parse_form_fields(form)

    async def apply(self, *, field_values: dict[str, str], resume: ResumeAttachment) -> ApplyResult:
        assert self.http_client is not None
        logs = [ApplyLogEntry(message="Fetching Greenhouse apply page", context={"url": self.job_url})]

        response, form = await self._fetch_form()
        action = form.get("action") or self.job_url
        submit_url = urljoin(str(response.url), str(action))

        resume_field = next(
            (name for name in _RESUME_FIELD_CANDIDATES if any(name == f.get("name") for f in form.find_all("input"))),
            "resume",
        )
        data, files = build_multipart_data(
            field_values=field_values, resume=resume, resume_field_name=resume_field
        )

        logs.append(ApplyLogEntry(message="Submitting application", context={"submit_url": submit_url}))
        submit_response = await self.http_client.post(submit_url, data=data, files=files)

        confirmation = detect_confirmation(text=submit_response.text, url=str(submit_response.url))
        success = submit_response.status_code < 400 and (
            confirmation.detected or submit_response.status_code in (200, 201, 302)
        )

        return ApplyResult(
            success=success,
            confirmation_detected=confirmation.detected,
            confirmation_text=confirmation.matched_phrase,
            confirmation_url=str(submit_response.url),
            submitted_field_values=data,
            error_message=None if success else f"Unexpected response status {submit_response.status_code}",
            logs=logs,
        )


__all__ = ["GreenhouseApplyAdapter"]
