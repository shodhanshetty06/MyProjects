"""Admin module — Pydantic request/response contracts.

Every list endpoint binds a `*Filters(PaginationParams)` subclass as a
query-param dependency (`filters: XFilters = Depends()`), the same
`Depends()`-bound filter-schema convention `analytics.schemas.AnalyticsFilters`
uses, extended with pagination (`app.schemas.common.PaginationParams`).
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pydantic import BaseModel, Field, model_validator

from app.modules.admin.models import (
    AdminRole,
    ModerationStatus,
    ReportStatus,
    ReportTargetType,
    UserAccountStatus,
    VerificationStatus,
)
from app.modules.admin.permissions import Permission
from app.modules.job_ingestion.models import JobStatus
from app.modules.tracking.models import ApplicationStatus
from app.schemas.common import PaginationParams


class _DateRangeFilters(PaginationParams):
    date_from: datetime | None = None
    date_to: datetime | None = None

    @model_validator(mode="after")
    def _validate_range(self) -> "_DateRangeFilters":
        if self.date_from is not None and self.date_to is not None and self.date_from > self.date_to:
            raise ValueError("date_from cannot be after date_to")
        return self


# ---------------------------------------------------------------------------
# Actors / Permissions
# ---------------------------------------------------------------------------


class AdminActorBootstrapRequest(BaseModel):
    email: str = Field(max_length=255)
    display_name: str | None = Field(default=None, max_length=255)


class AdminActorCreateRequest(BaseModel):
    user_id: uuid.UUID
    email: str = Field(max_length=255)
    display_name: str = Field(max_length=255)
    role: AdminRole = AdminRole.SUPPORT


class AdminActorUpdateRequest(BaseModel):
    display_name: str | None = Field(default=None, max_length=255)
    role: AdminRole | None = None
    is_active: bool | None = None


class AdminActorResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    email: str
    display_name: str
    role: AdminRole
    is_active: bool
    last_active_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class RolePermissions(BaseModel):
    role: AdminRole
    permissions: list[Permission]


class PermissionsResponse(BaseModel):
    current_role: AdminRole
    current_permissions: list[Permission]
    roles: list[RolePermissions]


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


class AuditLogEntry(BaseModel):
    id: uuid.UUID
    actor_id: uuid.UUID | None
    actor_email: str | None
    action: str
    target_type: str
    target_id: uuid.UUID | None
    before: dict | None
    after: dict | None
    notes: str | None
    created_at: datetime


class SystemLogEntry(BaseModel):
    source: str
    severity: str
    message: str
    target_id: uuid.UUID | None
    occurred_at: datetime


class DashboardResponse(BaseModel):
    generated_at: datetime
    users_total: int
    users_active: int
    users_suspended: int
    users_banned: int
    companies_total: int
    companies_verified: int
    companies_pending: int
    jobs_total: int
    jobs_active: int
    jobs_flagged: int
    applications_total: int
    reports_open: int
    feature_flags_enabled: int
    recent_audit_logs: list[AuditLogEntry]
    recent_system_logs: list[SystemLogEntry]


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


class UserFilters(PaginationParams):
    search: str | None = Field(default=None, max_length=255)
    status: UserAccountStatus | None = None


class UserSummary(BaseModel):
    user_id: uuid.UUID
    status: UserAccountStatus
    application_count: int
    first_seen_at: datetime | None
    last_activity_at: datetime | None


class UserDetail(UserSummary):
    reason: str | None
    changed_at: datetime | None
    applications_by_status: dict[str, int]
    auto_apply_task_count: int
    notification_count: int


class UserStatusUpdateRequest(BaseModel):
    status: UserAccountStatus
    reason: str | None = Field(default=None, max_length=1000)


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------


class CompanyFilters(PaginationParams):
    search: str | None = Field(default=None, max_length=255)
    verification_status: VerificationStatus | None = None
    is_active: bool | None = None


class CompanySummary(BaseModel):
    id: uuid.UUID
    name: str
    website: str | None
    industry: str | None
    is_active: bool
    verification_status: VerificationStatus
    jobs_count: int
    created_at: datetime


class CompanyDetail(CompanySummary):
    description: str | None
    logo_url: str | None
    notes: str | None
    reviewed_by: uuid.UUID | None
    reviewed_at: datetime | None


class CompanyVerificationUpdateRequest(BaseModel):
    status: VerificationStatus
    notes: str | None = Field(default=None, max_length=2000)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


class JobFilters(PaginationParams):
    search: str | None = Field(default=None, max_length=255)
    moderation_status: ModerationStatus | None = None
    job_status: JobStatus | None = None
    company_id: uuid.UUID | None = None


class JobSummary(BaseModel):
    id: uuid.UUID
    title: str
    company_id: uuid.UUID
    company_name: str
    status: JobStatus
    moderation_status: ModerationStatus
    flag_count: int
    posted_at: datetime | None
    created_at: datetime


class JobModerationUpdateRequest(BaseModel):
    status: ModerationStatus
    reason: str | None = Field(default=None, max_length=2000)


# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------


class ApplicationFilters(PaginationParams):
    search: str | None = Field(default=None, max_length=255)
    user_id: uuid.UUID | None = None
    status: ApplicationStatus | None = None


class ApplicationSummary(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    company_name: str
    job_title: str
    status: ApplicationStatus
    source: str
    applied_at: datetime


# ---------------------------------------------------------------------------
# Reports / Content Moderation
# ---------------------------------------------------------------------------


class ReportFilters(PaginationParams):
    status: ReportStatus | None = None
    target_type: ReportTargetType | None = None


class ReportCreateRequest(BaseModel):
    target_type: ReportTargetType
    target_id: uuid.UUID
    reason: str = Field(max_length=2000)


class ReportResponse(BaseModel):
    id: uuid.UUID
    reporter_user_id: uuid.UUID
    target_type: ReportTargetType
    target_id: uuid.UUID
    reason: str
    status: ReportStatus
    resolved_by: uuid.UUID | None
    resolution_notes: str | None
    resolved_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ReportUpdateRequest(BaseModel):
    status: ReportStatus
    resolution_notes: str | None = Field(default=None, max_length=2000)


class ModerationQueueItem(BaseModel):
    kind: str
    id: uuid.UUID
    target_type: str
    target_id: uuid.UUID
    summary: str
    status: str
    created_at: datetime


class ModerationQueueResponse(BaseModel):
    items: list[ModerationQueueItem]
    total_open: int


# ---------------------------------------------------------------------------
# Feature Flags
# ---------------------------------------------------------------------------


class FeatureFlagCreateRequest(BaseModel):
    key: str = Field(max_length=100)
    name: str = Field(max_length=255)
    description: str | None = Field(default=None, max_length=2000)
    is_enabled: bool = False
    rollout_percentage: int = Field(default=100, ge=0, le=100)


class FeatureFlagUpdateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=255)
    description: str | None = Field(default=None, max_length=2000)
    is_enabled: bool | None = None
    rollout_percentage: int | None = Field(default=None, ge=0, le=100)


class FeatureFlagResponse(BaseModel):
    id: uuid.UUID
    key: str
    name: str
    description: str | None
    is_enabled: bool
    rollout_percentage: int
    updated_by: uuid.UUID | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Audit Logs / System Logs
# ---------------------------------------------------------------------------


class AuditLogFilters(_DateRangeFilters):
    actor_id: uuid.UUID | None = None
    action: str | None = Field(default=None, max_length=100)
    target_type: str | None = Field(default=None, max_length=50)


class SystemLogFilters(_DateRangeFilters):
    source: str | None = Field(default=None, max_length=100)
    severity: str | None = Field(default=None, max_length=20)


# ---------------------------------------------------------------------------
# Platform Analytics
# ---------------------------------------------------------------------------


class PlatformTrendPoint(BaseModel):
    bucket: date
    count: int


class PlatformAnalyticsResponse(BaseModel):
    generated_at: datetime
    days: int
    active_users_trend: list[PlatformTrendPoint]
    applications_trend: list[PlatformTrendPoint]
    jobs_posted_trend: list[PlatformTrendPoint]
    companies_added_trend: list[PlatformTrendPoint]
    reports_filed_trend: list[PlatformTrendPoint]
    reports_resolved_count: int
    jobs_moderated_count: int
    companies_reviewed_count: int


__all__ = [
    "AdminActorBootstrapRequest",
    "AdminActorCreateRequest",
    "AdminActorUpdateRequest",
    "AdminActorResponse",
    "RolePermissions",
    "PermissionsResponse",
    "AuditLogEntry",
    "SystemLogEntry",
    "DashboardResponse",
    "UserFilters",
    "UserSummary",
    "UserDetail",
    "UserStatusUpdateRequest",
    "CompanyFilters",
    "CompanySummary",
    "CompanyDetail",
    "CompanyVerificationUpdateRequest",
    "JobFilters",
    "JobSummary",
    "JobModerationUpdateRequest",
    "ApplicationFilters",
    "ApplicationSummary",
    "ReportFilters",
    "ReportCreateRequest",
    "ReportResponse",
    "ReportUpdateRequest",
    "ModerationQueueItem",
    "ModerationQueueResponse",
    "FeatureFlagCreateRequest",
    "FeatureFlagUpdateRequest",
    "FeatureFlagResponse",
    "AuditLogFilters",
    "SystemLogFilters",
    "PlatformTrendPoint",
    "PlatformAnalyticsResponse",
]
