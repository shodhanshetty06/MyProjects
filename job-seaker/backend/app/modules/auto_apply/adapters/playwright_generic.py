"""Generic Playwright apply adapter — the fallback for companies with no
known ATS (`ATSPlatform.CUSTOM_HTML` / `UNKNOWN`).

Entirely driven by CSS selectors stored in `CompanyIntelligence.adapter_config`,
under an `apply` sub-key so it doesn't collide with `job_ingestion`'s
scraping selector keys:

    {
      "apply": {
        "url": "https://example.com/careers/123/apply",   # optional, defaults to the job URL
        "fields": {                                         # canonical_field_name -> CSS selector
          "first_name": "#first-name",
          "email": "#email",
          "resume": "input[type=file]"
        },
        "submit_selector": "button[type=submit]",
        "confirmation_text": "application received"        # optional extra confirmation phrase
      }
    }

Best-effort by nature, same philosophy as `job_ingestion.CustomHTMLAdapter`:
a missing/renamed selector degrades gracefully rather than aborting the run.
"""
from __future__ import annotations

from app.core.exceptions import ValidationError
from app.modules.auto_apply.adapters import browser
from app.modules.auto_apply.adapters.base import (
    ApplicationField,
    ApplyLogEntry,
    ApplyResult,
    BaseApplyAdapter,
    CapturedScreenshot,
    FieldType,
    ResumeAttachment,
)
from app.modules.auto_apply.confirmation import ConfirmationResult, detect_confirmation
from app.modules.job_ingestion.models import ATSPlatform

_RESUME_FIELD_NAME = "resume"


class PlaywrightApplyAdapter(BaseApplyAdapter):
    platform = ATSPlatform.CUSTOM_HTML

    @property
    def _apply_config(self) -> dict:
        return self.adapter_config.get("apply") or {}

    @property
    def _apply_url(self) -> str:
        return str(self._apply_config.get("url") or self.job_url)

    @property
    def _field_selectors(self) -> dict[str, str]:
        return dict(self._apply_config.get("fields") or {})

    async def discover_fields(self) -> list[ApplicationField]:
        selectors = self._field_selectors
        if not selectors:
            raise ValidationError(
                "Generic Playwright adapter requires `adapter_config.apply.fields`.",
                error_code="missing_apply_config",
            )
        return [
            ApplicationField(
                key=name,
                label=name.replace("_", " ").title(),
                field_type=FieldType.FILE if name == _RESUME_FIELD_NAME else FieldType.TEXT,
                required=name != _RESUME_FIELD_NAME,
                applicant_field_hint=name,
            )
            for name in selectors
        ]

    async def apply(self, *, field_values: dict[str, str], resume: ResumeAttachment) -> ApplyResult:
        config = self._apply_config
        selectors = self._field_selectors
        submit_selector = config.get("submit_selector")
        if not submit_selector:
            raise ValidationError(
                "Generic Playwright adapter requires `adapter_config.apply.submit_selector`.",
                error_code="missing_apply_config",
            )

        logs: list[ApplyLogEntry] = []
        screenshots: list[CapturedScreenshot] = []

        async with browser.browser_page() as page:
            await page.goto(self._apply_url, wait_until="domcontentloaded")
            logs.append(ApplyLogEntry(message="Opened application page", context={"url": self._apply_url}))

            resume_selector = selectors.get(_RESUME_FIELD_NAME)
            if resume_selector:
                await browser.upload_file_field(
                    page,
                    resume_selector,
                    filename=resume.filename,
                    content=resume.content,
                    mime_type=resume.content_type,
                )

            for name, selector in selectors.items():
                if name == _RESUME_FIELD_NAME:
                    continue
                value = field_values.get(name)
                if value is None:
                    continue
                if not await browser.fill_text_field(page, selector, value):
                    await browser.select_option_field(page, selector, value)

            screenshots.append(
                CapturedScreenshot(stage="before_submit", content=await browser.capture_screenshot(page))
            )

            if not await browser.click(page, submit_selector):
                return ApplyResult(
                    success=False,
                    error_message=f"Submit selector '{submit_selector}' was not found on the page.",
                    logs=logs,
                    screenshots=screenshots,
                )

            await page.wait_for_timeout(2000)
            page_text = await page.content()
            page_url = page.url
            screenshots.append(
                CapturedScreenshot(stage="after_submit", content=await browser.capture_screenshot(page))
            )

        extra_phrase = config.get("confirmation_text")
        confirmation = detect_confirmation(text=page_text, url=page_url)
        if not confirmation.detected and extra_phrase and extra_phrase.lower() in page_text.lower():
            confirmation = ConfirmationResult(
                detected=True, matched_phrase=extra_phrase, matched_via="text", confidence=0.9
            )

        return ApplyResult(
            success=confirmation.detected,
            confirmation_detected=confirmation.detected,
            confirmation_text=confirmation.matched_phrase,
            confirmation_url=page_url,
            submitted_field_values=field_values,
            error_message=None if confirmation.detected else "No confirmation detected after submission.",
            logs=logs,
            screenshots=screenshots,
        )


__all__ = ["PlaywrightApplyAdapter"]
