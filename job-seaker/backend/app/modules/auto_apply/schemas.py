"""Auto Apply module — Pydantic request/response contracts."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.modules.auto_apply.models import (
    AutoApplyAttemptStatus,
    AutoApplyTaskStatus,
    FailureCategory,
    LogLevel,
    ScreenshotStage,
)
from app.modules.job_ingestion.models import ATSPlatform


class ApplicantProfileUpsertRequest(BaseModel):
    auto_apply_enabled: bool = True

    first_name: str | None = Field(default=None, max_length=100)
    last_name: str | None = Field(default=None, max_length=100)
    email: str | None = Field(default=None, max_length=255)
    phone: str | None = Field(default=None, max_length=50)

    address_line1: str | None = Field(default=None, max_length=255)
    city: str | None = Field(default=None, max_length=255)
    state: str | None = Field(default=None, max_length=255)
    postal_code: str | None = Field(default=None, max_length=20)
    country: str | None = Field(default=None, max_length=255)

    linkedin_url: str | None = Field(default=None, max_length=512)
    portfolio_url: str | None = Field(default=None, max_length=512)
    github_url: str | None = Field(default=None, max_length=512)
    website_url: str | None = Field(default=None, max_length=512)

    work_authorized: bool | None = None
    requires_sponsorship: bool | None = None
    willing_to_relocate: bool | None = None
    years_of_experience: int | None = Field(default=None, ge=0, le=80)
    desired_salary: str | None = Field(default=None, max_length=100)
    notice_period: str | None = Field(default=None, max_length=100)

    gender: str | None = Field(default=None, max_length=100)
    race_ethnicity: str | None = Field(default=None, max_length=100)
    veteran_status: str | None = Field(default=None, max_length=100)
    disability_status: str | None = Field(default=None, max_length=100)

    default_ats_resume_id: uuid.UUID | None = None
    default_resume_id: uuid.UUID | None = None
    custom_answers: dict[str, str] = Field(default_factory=dict)


class ApplicantProfileRead(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    auto_apply_enabled: bool
    first_name: str | None
    last_name: str | None
    email: str | None
    phone: str | None
    address_line1: str | None
    city: str | None
    state: str | None
    postal_code: str | None
    country: str | None
    linkedin_url: str | None
    portfolio_url: str | None
    github_url: str | None
    website_url: str | None
    work_authorized: bool | None
    requires_sponsorship: bool | None
    willing_to_relocate: bool | None
    years_of_experience: int | None
    desired_salary: str | None
    notice_period: str | None
    gender: str | None
    race_ethnicity: str | None
    veteran_status: str | None
    disability_status: str | None
    default_ats_resume_id: uuid.UUID | None
    default_resume_id: uuid.UUID | None
    custom_answers: dict[str, str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AutoApplyTaskCreateRequest(BaseModel):
    job_id: uuid.UUID
    ats_resume_id: uuid.UUID | None = Field(
        default=None, description="Explicit tailored-resume override; auto-selected if unset."
    )
    resume_version_id: uuid.UUID | None = Field(
        default=None, description="Explicit plain-resume override; used if `ats_resume_id` is unset."
    )


class AutoApplyScreenshotRead(BaseModel):
    id: uuid.UUID
    stage: ScreenshotStage
    created_at: datetime

    model_config = {"from_attributes": True}


class AutoApplyLogEntryRead(BaseModel):
    id: uuid.UUID
    level: LogLevel
    message: str
    context: dict
    created_at: datetime

    model_config = {"from_attributes": True}


class AutoApplyAttemptSummary(BaseModel):
    id: uuid.UUID
    attempt_number: int
    status: AutoApplyAttemptStatus
    failure_category: FailureCategory | None
    started_at: datetime
    finished_at: datetime | None
    error_message: str | None

    model_config = {"from_attributes": True}


class AutoApplyAttemptDetail(AutoApplyAttemptSummary):
    screenshots: list[AutoApplyScreenshotRead] = Field(default_factory=list)
    logs: list[AutoApplyLogEntryRead] = Field(default_factory=list)


class AutoApplyTaskRead(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    job_id: uuid.UUID | None
    company_name: str
    job_title: str
    job_url: str
    platform: ATSPlatform
    status: AutoApplyTaskStatus
    attempt_count: int
    max_attempts: int
    resume_used_display_name: str | None
    missing_fields: list
    confirmation_detected: bool
    confirmation_text: str | None
    submitted_at: datetime | None
    tracking_application_id: uuid.UUID | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AutoApplyTaskDetail(AutoApplyTaskRead):
    field_mapping_snapshot: dict
    attempts: list[AutoApplyAttemptSummary] = Field(default_factory=list)


class AutoApplyTaskFilters(BaseModel):
    status: list[AutoApplyTaskStatus] | None = None
    job_id: uuid.UUID | None = None


class AutoApplyScreenshotDownloadResponse(BaseModel):
    download_url: str


__all__ = [
    "ApplicantProfileUpsertRequest",
    "ApplicantProfileRead",
    "AutoApplyTaskCreateRequest",
    "AutoApplyScreenshotRead",
    "AutoApplyLogEntryRead",
    "AutoApplyAttemptSummary",
    "AutoApplyAttemptDetail",
    "AutoApplyTaskRead",
    "AutoApplyTaskDetail",
    "AutoApplyTaskFilters",
    "AutoApplyScreenshotDownloadResponse",
]
