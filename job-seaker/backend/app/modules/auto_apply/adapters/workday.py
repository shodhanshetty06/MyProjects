"""Workday apply adapter.

Unlike Greenhouse/Lever, a Workday CXS apply flow is a genuine
multi-step Angular SPA (My Information -> My Experience -> Application
Questions -> Voluntary Disclosures -> Review) with no plain-form
equivalent, so this adapter drives a real headless browser via
`adapters/browser.py`.

`discover_fields()` only returns the well-known, near-universal first step
(contact info) — the rest of the flow is only knowable by actually walking
it, so later required fields that have no mapped value are reported via
`ApplyResult.missing_fields` instead, and the walk stops at the first step
it can't fill.
"""
from __future__ import annotations

from app.core.config import get_settings
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
from app.modules.auto_apply.confirmation import detect_confirmation
from app.modules.job_ingestion.models import ATSPlatform

# Workday's `data-automation-id` values are unusually stable across tenants
# (they come from Workday's own component library, not per-company
# customization) — this is the well-known "My Information" step.
_CONTACT_STEP_FIELDS: list[ApplicationField] = [
    ApplicationField(
        key="legalNameSection_firstName", label="First Name", required=True, applicant_field_hint="first_name"
    ),
    ApplicationField(
        key="legalNameSection_lastName", label="Last Name", required=True, applicant_field_hint="last_name"
    ),
    ApplicationField(key="email", label="Email Address", required=True, applicant_field_hint="email"),
    ApplicationField(
        key="phone-number", label="Phone Number", required=True, applicant_field_hint="phone",
        field_type=FieldType.PHONE,
    ),
    ApplicationField(
        key="addressSection_addressLine1", label="Address Line 1", required=True,
        applicant_field_hint="address_line1",
    ),
    ApplicationField(key="addressSection_city", label="City", required=True, applicant_field_hint="city"),
    ApplicationField(
        key="addressSection_postalCode", label="Postal Code", required=True,
        applicant_field_hint="postal_code",
    ),
]

_SELECTOR_BY_KEY = {field.key: f'[data-automation-id="{field.key}"]' for field in _CONTACT_STEP_FIELDS}
_RESUME_UPLOAD_SELECTOR = '[data-automation-id="file-upload-input-ref"]'
_NEXT_BUTTON_SELECTOR = '[data-automation-id="bottom-navigation-next-button"]'
_SUBMIT_BUTTON_SELECTOR = '[data-automation-id="submit-button"], [data-automation-id="wd-Submit"]'


class WorkdayApplyAdapter(BaseApplyAdapter):
    platform = ATSPlatform.WORKDAY

    async def discover_fields(self) -> list[ApplicationField]:
        return list(_CONTACT_STEP_FIELDS)

    async def apply(self, *, field_values: dict[str, str], resume: ResumeAttachment) -> ApplyResult:
        settings = get_settings()
        logs: list[ApplyLogEntry] = []
        screenshots: list[CapturedScreenshot] = []
        missing: list[ApplicationField] = []

        async with browser.browser_page() as page:
            await page.goto(self.job_url, wait_until="domcontentloaded")
            logs.append(ApplyLogEntry(message="Opened Workday job page", context={"url": self.job_url}))

            await browser.click(page, '[data-automation-id="apply"], [data-automation-id="applyManually"]')
            await browser.upload_file_field(
                page,
                _RESUME_UPLOAD_SELECTOR,
                filename=resume.filename,
                content=resume.content,
                mime_type=resume.content_type,
            )

            for step in range(settings.AUTO_APPLY_WORKDAY_MAX_STEPS):
                screenshots.append(
                    CapturedScreenshot(stage="step", content=await browser.capture_screenshot(page))
                )

                for field in _CONTACT_STEP_FIELDS:
                    selector = _SELECTOR_BY_KEY[field.key]
                    value = field_values.get(field.key)
                    if value is None:
                        if await browser.element_exists(page, selector):
                            missing.append(field)
                        continue
                    await browser.fill_text_field(page, selector, value)

                if missing:
                    logs.append(
                        ApplyLogEntry(
                            level="warning",
                            message="Required Workday field(s) have no mapped value",
                            context={"fields": [f.key for f in missing]},
                        )
                    )
                    return ApplyResult(
                        success=False,
                        missing_fields=missing,
                        logs=logs,
                        screenshots=screenshots,
                        error_message="Workday application requires additional profile data.",
                    )

                if await browser.element_exists(page, _SUBMIT_BUTTON_SELECTOR):
                    await browser.click(page, _SUBMIT_BUTTON_SELECTOR)
                    break
                if not await browser.click(page, _NEXT_BUTTON_SELECTOR):
                    logs.append(
                        ApplyLogEntry(level="warning", message="No further Workday step navigation found")
                    )
                    break

            await page.wait_for_timeout(settings.AUTO_APPLY_CONFIRMATION_WAIT_MS)
            page_text = await page.content()
            page_url = page.url
            screenshots.append(
                CapturedScreenshot(stage="after_submit", content=await browser.capture_screenshot(page))
            )

        confirmation = detect_confirmation(text=page_text, url=page_url)
        logs.append(
            ApplyLogEntry(
                message="Workday submission finished",
                context={"confirmation_detected": confirmation.detected},
            )
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


__all__ = ["WorkdayApplyAdapter"]
