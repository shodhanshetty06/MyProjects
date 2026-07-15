"""Apply adapter base contract.

Mirrors `job_ingestion.adapters.base` — one `BaseApplyAdapter` subclass per
ATS platform, all returning the same `ApplyResult` DTO regardless of
whether the adapter drives a real browser (Workday, the generic Playwright
fallback) or talks HTTP directly to a plain HTML form (Greenhouse, Lever).
Everything downstream (`field_mapping.py`, `tasks.py`) only ever depends on
`ApplicationField` / `ApplyResult`, never on adapter internals.
"""
from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel, Field

from app.modules.job_ingestion.models import ATSPlatform


class FieldType(str, enum.Enum):
    TEXT = "text"
    TEXTAREA = "textarea"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    SELECT = "select"
    MULTISELECT = "multiselect"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    FILE = "file"
    DATE = "date"


class ApplicationField(BaseModel):
    """One discovered form field/question on an ATS application form."""

    key: str = Field(description="Adapter-local identifier (form field name / question id).")
    label: str = Field(description="Human-readable question/label text.")
    field_type: FieldType = FieldType.TEXT
    required: bool = False
    options: list[str] = Field(default_factory=list, description="Choices, for select/radio fields.")
    applicant_field_hint: str | None = Field(
        default=None,
        description=(
            "Canonical ApplicantProfile field name, if the adapter can resolve it directly "
            "(e.g. a Greenhouse `#first_name` input) — leave unset for free-form questions "
            "that `field_mapping.py` must resolve by matching `label`."
        ),
    )


class ApplyLogEntry(BaseModel):
    level: str = "info"
    message: str
    context: dict[str, Any] = Field(default_factory=dict)


class CapturedScreenshot(BaseModel):
    stage: str
    content: bytes

    model_config = {"arbitrary_types_allowed": True}


class ResumeAttachment(BaseModel):
    """Resolved resume file to attach to the application."""

    filename: str
    content_type: str
    content: bytes

    model_config = {"arbitrary_types_allowed": True}


class ApplyResult(BaseModel):
    success: bool
    confirmation_detected: bool = False
    confirmation_text: str | None = None
    confirmation_url: str | None = None
    submitted_field_values: dict[str, str] = Field(default_factory=dict)
    error_message: str | None = None
    # Additional required fields discovered only while actually walking a
    # multi-step flow (e.g. Workday's later steps) — merged into the task's
    # `missing_fields` alongside whatever `discover_fields()` found upfront.
    missing_fields: list[ApplicationField] = Field(default_factory=list)
    logs: list[ApplyLogEntry] = Field(default_factory=list)
    screenshots: list[CapturedScreenshot] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class BaseApplyAdapter(ABC):
    """Common interface every concrete apply adapter implements."""

    platform: ClassVar[ATSPlatform]

    def __init__(
        self,
        *,
        job_url: str,
        external_id: str,
        ats_identifier: str | None,
        adapter_config: dict[str, Any],
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.job_url = job_url
        self.external_id = external_id
        self.ats_identifier = ats_identifier
        self.adapter_config = adapter_config
        self.http_client = http_client

    @abstractmethod
    async def discover_fields(self) -> list[ApplicationField]:
        """Best-effort upfront discovery of the application form's fields.

        Used by `field_mapping.py` to compute missing required fields
        *before* attempting a submission. Adapters whose real form is only
        knowable by actually walking the flow (Workday) may return a
        partial/first-step-only list here and report the rest via
        `ApplyResult.missing_fields`.
        """
        raise NotImplementedError

    @abstractmethod
    async def apply(
        self, *, field_values: dict[str, str], resume: ResumeAttachment
    ) -> ApplyResult:
        """Fill and submit the application. Must never raise for expected,
        classifiable failure modes (missing selectors, HTTP errors) —
        callers rely on `ApplyResult.success` / `error_message` for those.
        Unexpected exceptions still propagate so `tasks.py` can classify
        and retry them.
        """
        raise NotImplementedError


__all__ = [
    "FieldType",
    "ApplicationField",
    "ApplyLogEntry",
    "CapturedScreenshot",
    "ResumeAttachment",
    "ApplyResult",
    "BaseApplyAdapter",
]
