"""Lever apply adapter.

Lever's hosted apply page (`{hostedUrl}/apply`) is, like Greenhouse's, a
plain server-rendered `<form>` — handled the same HTTP-only way as
`greenhouse.py`, sharing the `html_form` parser.
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


class LeverApplyAdapter(BaseApplyAdapter):
    platform = ATSPlatform.LEVER

    @property
    def _apply_url(self) -> str:
        base = self.job_url.rstrip("/")
        return base if base.endswith("/apply") else f"{base}/apply"

    async def _fetch_form(self):
        assert self.http_client is not None
        try:
            response = await self.http_client.get(self._apply_url)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 -- reclassified by errors.classify_exception upstream
            raise ServiceUnavailableError(f"Could not load the Lever apply page: {exc}") from exc

        form = find_form(response.text, form_selector="form.application-form, form")
        if form is None:
            raise ServiceUnavailableError("Lever apply page has no recognizable application form.")
        return response, form

    async def discover_fields(self) -> list[ApplicationField]:
        _, form = await self._fetch_form()
        return parse_form_fields(form)

    async def apply(self, *, field_values: dict[str, str], resume: ResumeAttachment) -> ApplyResult:
        assert self.http_client is not None
        logs = [ApplyLogEntry(message="Fetching Lever apply page", context={"url": self._apply_url})]

        response, form = await self._fetch_form()
        action = form.get("action") or self._apply_url
        submit_url = urljoin(str(response.url), str(action))

        data, files = build_multipart_data(field_values=field_values, resume=resume, resume_field_name="resume")

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


__all__ = ["LeverApplyAdapter"]
