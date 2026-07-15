"""Tracking module — Pydantic request/response contracts."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.modules.tracking.models import ApplicationSource, ApplicationStatus


class ApplicationSortBy(str, enum.Enum):
    APPLIED_AT = "applied_at"
    UPDATED_AT = "updated_at"
    COMPANY_NAME = "company_name"
    JOB_TITLE = "job_title"
    STATUS = "status"


class SortDirection(str, enum.Enum):
    ASC = "asc"
    DESC = "desc"


class ApplicationFilters(BaseModel):
    """Bound as a query-param dependency: `filters: ApplicationFilters = Depends()`.

    Pagination is intentionally not part of this model — reuse
    `app.schemas.common.PaginationParams` separately, matching how
    search/job_ingestion do it.
    """

    q: str | None = Field(default=None, max_length=255)
    status: list[ApplicationStatus] | None = None
    company_name: str | None = Field(default=None, max_length=255)
    job_id: uuid.UUID | None = None
    applied_after: datetime | None = None
    applied_before: datetime | None = None

    sort_by: ApplicationSortBy = ApplicationSortBy.APPLIED_AT
    sort_direction: SortDirection = SortDirection.DESC

    @model_validator(mode="after")
    def _validate(self) -> "ApplicationFilters":
        if (
            self.applied_after is not None
            and self.applied_before is not None
            and self.applied_after > self.applied_before
        ):
            raise ValueError("applied_after cannot be after applied_before")
        return self


class ApplicationCreateRequest(BaseModel):
    job_id: uuid.UUID | None = None
    company_name: str = Field(min_length=1, max_length=255)
    job_title: str = Field(min_length=1, max_length=512)
    job_url: str | None = Field(default=None, max_length=1024)
    location: str | None = Field(default=None, max_length=512)
    source: ApplicationSource = ApplicationSource.MANUAL
    status: ApplicationStatus = ApplicationStatus.APPLIED
    applied_at: datetime | None = None


class ApplicationUpdateRequest(BaseModel):
    company_name: str | None = Field(default=None, min_length=1, max_length=255)
    job_title: str | None = Field(default=None, min_length=1, max_length=512)
    job_url: str | None = Field(default=None, max_length=1024)
    location: str | None = Field(default=None, max_length=512)
    applied_at: datetime | None = None


class ApplicationStatusChangeRequest(BaseModel):
    status: ApplicationStatus
    note: str | None = Field(default=None, max_length=2000)


class ApplicationNoteCreateRequest(BaseModel):
    content: str = Field(min_length=1, max_length=5000)


class ApplicationNoteUpdateRequest(BaseModel):
    content: str = Field(min_length=1, max_length=5000)


class ApplicationNoteRead(BaseModel):
    id: uuid.UUID
    content: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ApplicationStatusHistoryRead(BaseModel):
    id: uuid.UUID
    from_status: ApplicationStatus | None
    to_status: ApplicationStatus
    note: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ApplicationRead(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    job_id: uuid.UUID | None
    company_name: str
    job_title: str
    job_url: str | None
    location: str | None
    status: ApplicationStatus
    source: ApplicationSource
    applied_at: datetime
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ApplicationDetail(ApplicationRead):
    notes: list[ApplicationNoteRead] = Field(default_factory=list)
    status_history: list[ApplicationStatusHistoryRead] = Field(default_factory=list)


class TimelineEntryType(str, enum.Enum):
    STATUS_CHANGE = "status_change"
    NOTE = "note"


class TimelineEntry(BaseModel):
    type: TimelineEntryType
    occurred_at: datetime
    data: dict[str, Any]


class ApplicationStatistics(BaseModel):
    total_applications: int
    status_counts: dict[ApplicationStatus, int]
    applications_last_7_days: int
    applications_last_30_days: int
    interview_rate: float
    offer_rate: float
    acceptance_rate: float
    average_days_to_first_response: float | None


class DashboardResponse(BaseModel):
    statistics: ApplicationStatistics
    recent_applications: list[ApplicationRead]
    needs_follow_up: list[ApplicationRead]


__all__ = [
    "ApplicationSortBy",
    "SortDirection",
    "ApplicationFilters",
    "ApplicationCreateRequest",
    "ApplicationUpdateRequest",
    "ApplicationStatusChangeRequest",
    "ApplicationNoteCreateRequest",
    "ApplicationNoteUpdateRequest",
    "ApplicationNoteRead",
    "ApplicationStatusHistoryRead",
    "ApplicationRead",
    "ApplicationDetail",
    "TimelineEntryType",
    "TimelineEntry",
    "ApplicationStatistics",
    "DashboardResponse",
]
