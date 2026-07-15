"""Admin module — API routes (v1).

Every route requires a verified JWT identity (`CurrentUserIdDep`, see
`app/api/deps.py`) like every other module. Routes that act *as* the Admin
Portal additionally resolve an `AdminActor` for that user
(`deps.CurrentAdminActorDep`) and, where relevant, enforce a specific
`Permission` via `deps.require_permission`. `POST /reports` is the one
exception — any platform user can file a report, not just admin staff.
"""
from __future__ import annotations

import math
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from app.api.deps import CurrentUserIdDep, DBSessionDep
from app.modules.admin.deps import CurrentAdminActorDep, require_permission
from app.modules.admin.models import AdminActor
from app.modules.admin.permissions import Permission
from app.modules.admin.repository import AdminRepository
from app.modules.admin.schemas import (
    AdminActorBootstrapRequest,
    AdminActorCreateRequest,
    AdminActorResponse,
    AdminActorUpdateRequest,
    ApplicationFilters,
    ApplicationSummary,
    AuditLogEntry,
    AuditLogFilters,
    CompanyDetail,
    CompanyFilters,
    CompanySummary,
    CompanyVerificationUpdateRequest,
    DashboardResponse,
    FeatureFlagCreateRequest,
    FeatureFlagResponse,
    FeatureFlagUpdateRequest,
    JobFilters,
    JobModerationUpdateRequest,
    JobSummary,
    ModerationQueueResponse,
    PermissionsResponse,
    PlatformAnalyticsResponse,
    ReportCreateRequest,
    ReportFilters,
    ReportResponse,
    ReportUpdateRequest,
    SystemLogEntry,
    SystemLogFilters,
    UserDetail,
    UserFilters,
    UserStatusUpdateRequest,
    UserSummary,
)
from app.modules.admin.service import AdminService
from app.schemas.common import BaseResponse, PaginatedResponse, PaginationMeta

router = APIRouter(prefix="/admin", tags=["Admin"])


def get_admin_service(session: DBSessionDep) -> AdminService:
    return AdminService(AdminRepository(session))


ServiceDep = Depends(get_admin_service)


def _paginated(items: list, total: int, page: int, page_size: int) -> PaginatedResponse:
    return PaginatedResponse(
        data=items,
        pagination=PaginationMeta(
            page=page,
            page_size=page_size,
            total_items=total,
            total_pages=math.ceil(total / page_size) if page_size else 0,
        ),
    )


# ---------------------------------------------------------------------------
# Actors / Permissions
# ---------------------------------------------------------------------------


@router.post("/actors/bootstrap", response_model=BaseResponse[AdminActorResponse], status_code=status.HTTP_201_CREATED)
async def bootstrap_actor(
    user_id: CurrentUserIdDep,
    request: AdminActorBootstrapRequest,
    service: AdminService = ServiceDep,
) -> BaseResponse[AdminActorResponse]:
    actor = await service.bootstrap_actor(user_id, request)
    return BaseResponse(message="Admin actor bootstrapped as SUPER_ADMIN.", data=actor)


@router.get("/actors", response_model=PaginatedResponse[AdminActorResponse])
async def list_actors(
    actor: Annotated[AdminActor, require_permission(Permission.PERMISSIONS_MANAGE)],
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    service: AdminService = ServiceDep,
) -> PaginatedResponse[AdminActorResponse]:
    items, total = await service.list_actors(page=page, page_size=page_size)
    return _paginated(items, total, page, page_size)


@router.post("/actors", response_model=BaseResponse[AdminActorResponse], status_code=status.HTTP_201_CREATED)
async def create_actor(
    actor: Annotated[AdminActor, require_permission(Permission.PERMISSIONS_MANAGE)],
    request: AdminActorCreateRequest,
    service: AdminService = ServiceDep,
) -> BaseResponse[AdminActorResponse]:
    created = await service.create_actor(actor, request)
    return BaseResponse(data=created)


@router.patch("/actors/{actor_id}", response_model=BaseResponse[AdminActorResponse])
async def update_actor(
    actor_id: uuid.UUID,
    actor: Annotated[AdminActor, require_permission(Permission.PERMISSIONS_MANAGE)],
    request: AdminActorUpdateRequest,
    service: AdminService = ServiceDep,
) -> BaseResponse[AdminActorResponse]:
    updated = await service.update_actor(actor, actor_id, request)
    return BaseResponse(data=updated)


@router.get("/permissions", response_model=BaseResponse[PermissionsResponse])
async def get_permissions(
    actor: CurrentAdminActorDep, service: AdminService = ServiceDep
) -> BaseResponse[PermissionsResponse]:
    return BaseResponse(data=await service.get_permissions(actor))


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@router.get("/dashboard", response_model=BaseResponse[DashboardResponse])
async def get_dashboard(
    actor: Annotated[AdminActor, require_permission(Permission.DASHBOARD_VIEW)],
    service: AdminService = ServiceDep,
) -> BaseResponse[DashboardResponse]:
    return BaseResponse(data=await service.get_dashboard())


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

UserFiltersDep = Annotated[UserFilters, Depends()]


@router.get("/users", response_model=PaginatedResponse[UserSummary])
async def list_users(
    actor: Annotated[AdminActor, require_permission(Permission.USERS_MANAGE)],
    filters: UserFiltersDep,
    service: AdminService = ServiceDep,
) -> PaginatedResponse[UserSummary]:
    items, total = await service.list_users(filters)
    return _paginated(items, total, filters.page, filters.page_size)


@router.get("/users/{user_id}", response_model=BaseResponse[UserDetail])
async def get_user(
    user_id: uuid.UUID,
    actor: Annotated[AdminActor, require_permission(Permission.USERS_MANAGE)],
    service: AdminService = ServiceDep,
) -> BaseResponse[UserDetail]:
    return BaseResponse(data=await service.get_user(user_id))


@router.patch("/users/{user_id}/status", response_model=BaseResponse[UserDetail])
async def update_user_status(
    user_id: uuid.UUID,
    actor: Annotated[AdminActor, require_permission(Permission.USERS_MANAGE)],
    request: UserStatusUpdateRequest,
    service: AdminService = ServiceDep,
) -> BaseResponse[UserDetail]:
    return BaseResponse(data=await service.update_user_status(actor, user_id, request))


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------

CompanyFiltersDep = Annotated[CompanyFilters, Depends()]


@router.get("/companies", response_model=PaginatedResponse[CompanySummary])
async def list_companies(
    actor: Annotated[AdminActor, require_permission(Permission.DASHBOARD_VIEW)],
    filters: CompanyFiltersDep,
    service: AdminService = ServiceDep,
) -> PaginatedResponse[CompanySummary]:
    items, total = await service.list_companies(filters)
    return _paginated(items, total, filters.page, filters.page_size)


@router.get("/companies/{company_id}", response_model=BaseResponse[CompanyDetail])
async def get_company(
    company_id: uuid.UUID,
    actor: Annotated[AdminActor, require_permission(Permission.DASHBOARD_VIEW)],
    service: AdminService = ServiceDep,
) -> BaseResponse[CompanyDetail]:
    return BaseResponse(data=await service.get_company(company_id))


@router.patch("/companies/{company_id}/verification", response_model=BaseResponse[CompanyDetail])
async def update_company_verification(
    company_id: uuid.UUID,
    actor: Annotated[AdminActor, require_permission(Permission.COMPANIES_VERIFY)],
    request: CompanyVerificationUpdateRequest,
    service: AdminService = ServiceDep,
) -> BaseResponse[CompanyDetail]:
    return BaseResponse(data=await service.update_company_verification(actor, company_id, request))


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

JobFiltersDep = Annotated[JobFilters, Depends()]


@router.get("/jobs", response_model=PaginatedResponse[JobSummary])
async def list_jobs(
    actor: Annotated[AdminActor, require_permission(Permission.DASHBOARD_VIEW)],
    filters: JobFiltersDep,
    service: AdminService = ServiceDep,
) -> PaginatedResponse[JobSummary]:
    items, total = await service.list_jobs(filters)
    return _paginated(items, total, filters.page, filters.page_size)


@router.patch("/jobs/{job_id}/moderation", response_model=BaseResponse[JobSummary])
async def update_job_moderation(
    job_id: uuid.UUID,
    actor: Annotated[AdminActor, require_permission(Permission.JOBS_MODERATE)],
    request: JobModerationUpdateRequest,
    service: AdminService = ServiceDep,
) -> BaseResponse[JobSummary]:
    return BaseResponse(data=await service.update_job_moderation(actor, job_id, request))


# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------

ApplicationFiltersDep = Annotated[ApplicationFilters, Depends()]


@router.get("/applications", response_model=PaginatedResponse[ApplicationSummary])
async def list_applications(
    actor: Annotated[AdminActor, require_permission(Permission.USERS_MANAGE)],
    filters: ApplicationFiltersDep,
    service: AdminService = ServiceDep,
) -> PaginatedResponse[ApplicationSummary]:
    items, total = await service.list_applications(filters)
    return _paginated(items, total, filters.page, filters.page_size)


@router.delete("/applications/{application_id}", response_model=BaseResponse[None])
async def delete_application(
    application_id: uuid.UUID,
    actor: Annotated[AdminActor, require_permission(Permission.USERS_MANAGE)],
    service: AdminService = ServiceDep,
) -> BaseResponse[None]:
    await service.delete_application(actor, application_id)
    return BaseResponse(message="Application removed.")


# ---------------------------------------------------------------------------
# Reports / Content Moderation
# ---------------------------------------------------------------------------

ReportFiltersDep = Annotated[ReportFilters, Depends()]


@router.post("/reports", response_model=BaseResponse[ReportResponse], status_code=status.HTTP_201_CREATED)
async def create_report(
    user_id: CurrentUserIdDep,
    request: ReportCreateRequest,
    service: AdminService = ServiceDep,
) -> BaseResponse[ReportResponse]:
    return BaseResponse(data=await service.create_report(user_id, request))


@router.get("/reports", response_model=PaginatedResponse[ReportResponse])
async def list_reports(
    actor: Annotated[AdminActor, require_permission(Permission.CONTENT_MODERATE)],
    filters: ReportFiltersDep,
    service: AdminService = ServiceDep,
) -> PaginatedResponse[ReportResponse]:
    items, total = await service.list_reports(filters)
    return _paginated(items, total, filters.page, filters.page_size)


@router.patch("/reports/{report_id}", response_model=BaseResponse[ReportResponse])
async def update_report(
    report_id: uuid.UUID,
    actor: Annotated[AdminActor, require_permission(Permission.CONTENT_MODERATE)],
    request: ReportUpdateRequest,
    service: AdminService = ServiceDep,
) -> BaseResponse[ReportResponse]:
    return BaseResponse(data=await service.update_report(actor, report_id, request))


@router.get("/moderation/queue", response_model=BaseResponse[ModerationQueueResponse])
async def get_moderation_queue(
    actor: Annotated[AdminActor, require_permission(Permission.CONTENT_MODERATE)],
    service: AdminService = ServiceDep,
) -> BaseResponse[ModerationQueueResponse]:
    return BaseResponse(data=await service.get_moderation_queue())


# ---------------------------------------------------------------------------
# Feature Flags
# ---------------------------------------------------------------------------


@router.get("/feature-flags", response_model=BaseResponse[list[FeatureFlagResponse]])
async def list_feature_flags(
    actor: Annotated[AdminActor, require_permission(Permission.DASHBOARD_VIEW)],
    service: AdminService = ServiceDep,
) -> BaseResponse[list[FeatureFlagResponse]]:
    return BaseResponse(data=await service.list_feature_flags())


@router.post("/feature-flags", response_model=BaseResponse[FeatureFlagResponse], status_code=status.HTTP_201_CREATED)
async def create_feature_flag(
    actor: Annotated[AdminActor, require_permission(Permission.FEATURE_FLAGS_MANAGE)],
    request: FeatureFlagCreateRequest,
    service: AdminService = ServiceDep,
) -> BaseResponse[FeatureFlagResponse]:
    return BaseResponse(data=await service.create_feature_flag(actor, request))


@router.patch("/feature-flags/{key}", response_model=BaseResponse[FeatureFlagResponse])
async def update_feature_flag(
    key: str,
    actor: Annotated[AdminActor, require_permission(Permission.FEATURE_FLAGS_MANAGE)],
    request: FeatureFlagUpdateRequest,
    service: AdminService = ServiceDep,
) -> BaseResponse[FeatureFlagResponse]:
    return BaseResponse(data=await service.update_feature_flag(actor, key, request))


@router.delete("/feature-flags/{key}", response_model=BaseResponse[None])
async def delete_feature_flag(
    key: str,
    actor: Annotated[AdminActor, require_permission(Permission.FEATURE_FLAGS_MANAGE)],
    service: AdminService = ServiceDep,
) -> BaseResponse[None]:
    await service.delete_feature_flag(actor, key)
    return BaseResponse(message="Feature flag deleted.")


# ---------------------------------------------------------------------------
# Audit Logs / System Logs
# ---------------------------------------------------------------------------

AuditLogFiltersDep = Annotated[AuditLogFilters, Depends()]
SystemLogFiltersDep = Annotated[SystemLogFilters, Depends()]


@router.get("/audit-logs", response_model=PaginatedResponse[AuditLogEntry])
async def list_audit_logs(
    actor: Annotated[AdminActor, require_permission(Permission.AUDIT_LOGS_VIEW)],
    filters: AuditLogFiltersDep,
    service: AdminService = ServiceDep,
) -> PaginatedResponse[AuditLogEntry]:
    items, total = await service.list_audit_logs(filters)
    return _paginated(items, total, filters.page, filters.page_size)


@router.get("/system-logs", response_model=PaginatedResponse[SystemLogEntry])
async def list_system_logs(
    actor: Annotated[AdminActor, require_permission(Permission.SYSTEM_LOGS_VIEW)],
    filters: SystemLogFiltersDep,
    service: AdminService = ServiceDep,
) -> PaginatedResponse[SystemLogEntry]:
    items, total = await service.list_system_logs(filters)
    return _paginated(items, total, filters.page, filters.page_size)


# ---------------------------------------------------------------------------
# Platform Analytics
# ---------------------------------------------------------------------------


@router.get("/analytics", response_model=BaseResponse[PlatformAnalyticsResponse])
async def get_platform_analytics(
    actor: Annotated[AdminActor, require_permission(Permission.ANALYTICS_VIEW)],
    days: int = Query(default=30, ge=1, le=365),
    service: AdminService = ServiceDep,
) -> BaseResponse[PlatformAnalyticsResponse]:
    return BaseResponse(data=await service.get_platform_analytics(days))
