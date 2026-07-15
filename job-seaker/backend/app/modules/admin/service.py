"""Admin module — application/use-case layer.

Every mutating method writes exactly one `AuditLog` row (via `_audit`) inside
the same request — the audit-logging backbone for the whole portal, not a
bolt-on. Permission enforcement itself happens at the router layer
(`deps.require_permission`); methods here receive the already-authorized
`AdminActor` purely for audit attribution.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from app.core.exceptions import ConflictError, NotFoundError
from app.core.logging import get_logger
from app.modules.admin.models import (
    AdminActor,
    AdminRole,
    ModerationStatus,
    ReportStatus,
    UserAccountStatus,
    VerificationStatus,
)
from app.modules.admin.permissions import Permission, permissions_for_role
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
    ModerationQueueItem,
    ModerationQueueResponse,
    PermissionsResponse,
    PlatformAnalyticsResponse,
    PlatformTrendPoint,
    ReportCreateRequest,
    ReportFilters,
    ReportResponse,
    ReportUpdateRequest,
    RolePermissions,
    SystemLogFilters,
    UserDetail,
    UserFilters,
    UserStatusUpdateRequest,
    UserSummary,
)
from app.modules.admin.system_logs import collect_system_logs

logger = get_logger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AdminService:
    """Business logic orchestration for the Admin module."""

    def __init__(self, repository: AdminRepository) -> None:
        self.repository = repository

    async def _audit(
        self,
        actor: AdminActor | None,
        action: str,
        target_type: str,
        target_id: uuid.UUID | None,
        *,
        before: dict | None = None,
        after: dict | None = None,
        notes: str | None = None,
    ) -> None:
        await self.repository.create_audit_log(
            actor_id=actor.id if actor else None,
            action=action,
            target_type=target_type,
            target_id=target_id,
            before=before,
            after=after,
            notes=notes,
        )

    # --- actors / permissions --------------------------------------------------

    async def bootstrap_actor(
        self, user_id: uuid.UUID, request: AdminActorBootstrapRequest
    ) -> AdminActorResponse:
        # Serializes concurrent first-caller-wins bootstrap attempts so the
        # count-check below can't race two requests into both becoming
        # SUPER_ADMIN — see AdminRepository.acquire_bootstrap_lock.
        await self.repository.acquire_bootstrap_lock()
        if await self.repository.count_actors() > 0:
            raise ConflictError(
                "An admin actor already exists; bootstrap is only available once. "
                "Ask an existing SUPER_ADMIN/ADMIN to create your account."
            )
        actor = await self.repository.create_actor(
            user_id=user_id,
            email=request.email,
            display_name=request.display_name or request.email,
            role=AdminRole.SUPER_ADMIN,
        )
        await self._audit(
            actor, "actor.bootstrap", "admin_actor", actor.id, after={"role": actor.role}
        )
        return AdminActorResponse.model_validate(actor)

    async def list_actors(self, *, page: int, page_size: int) -> tuple[list[AdminActorResponse], int]:
        total = await self.repository.count_actors()
        rows = await self.repository.list_actors(offset=(page - 1) * page_size, limit=page_size)
        return [AdminActorResponse.model_validate(row) for row in rows], total

    async def create_actor(
        self, actor: AdminActor, request: AdminActorCreateRequest
    ) -> AdminActorResponse:
        if await self.repository.get_actor_by_user_id(request.user_id) is not None:
            raise ConflictError("An admin actor already exists for this user_id.")
        new_actor = await self.repository.create_actor(
            user_id=request.user_id,
            email=request.email,
            display_name=request.display_name,
            role=request.role,
        )
        await self._audit(
            actor,
            "actor.create",
            "admin_actor",
            new_actor.id,
            after={"role": new_actor.role, "email": new_actor.email},
        )
        return AdminActorResponse.model_validate(new_actor)

    async def update_actor(
        self, actor: AdminActor, target_id: uuid.UUID, request: AdminActorUpdateRequest
    ) -> AdminActorResponse:
        target = await self.repository.get_actor(target_id)
        if target is None:
            raise NotFoundError("Admin actor not found.")
        before = {"role": target.role, "is_active": target.is_active}
        updates = request.model_dump(exclude_unset=True)
        target = await self.repository.update_actor(target, **updates)
        await self._audit(
            actor,
            "actor.update",
            "admin_actor",
            target.id,
            before=before,
            after={"role": target.role, "is_active": target.is_active},
        )
        return AdminActorResponse.model_validate(target)

    async def get_permissions(self, actor: AdminActor) -> PermissionsResponse:
        return PermissionsResponse(
            current_role=actor.role,
            current_permissions=sorted(permissions_for_role(actor.role), key=lambda p: p.value),
            roles=[
                RolePermissions(role=role, permissions=sorted(perms, key=lambda p: p.value))
                for role, perms in (
                    (role, permissions_for_role(role)) for role in AdminRole
                )
            ],
        )

    # --- dashboard ---------------------------------------------------------------

    async def get_dashboard(self) -> DashboardResponse:
        user_overrides = await self.repository.user_counts_by_status()
        users_total = await self.repository.distinct_user_count()
        suspended = user_overrides.get(UserAccountStatus.SUSPENDED.value, 0)
        banned = user_overrides.get(UserAccountStatus.BANNED.value, 0)
        active = max(users_total - suspended - banned, 0)

        company_counts = await self.repository.company_verification_counts()
        job_counts = await self.repository.job_moderation_counts()
        applications_total = await self.repository.applications_total()
        reports_open = await self.repository.open_reports_count()
        feature_flags_enabled = await self.repository.enabled_feature_flags_count()

        recent_audit_logs = await self._build_audit_entries(
            await self.repository.recent_audit_logs(10)
        )
        recent_system_logs, _ = await collect_system_logs(
            self.repository.session, SystemLogFilters(page=1, page_size=10)
        )

        return DashboardResponse(
            generated_at=_now(),
            users_total=users_total,
            users_active=active,
            users_suspended=suspended,
            users_banned=banned,
            companies_total=company_counts.get("total", 0),
            companies_verified=company_counts.get(VerificationStatus.VERIFIED.value, 0),
            companies_pending=company_counts.get(VerificationStatus.PENDING.value, 0),
            jobs_total=job_counts.get("total", 0),
            jobs_active=job_counts.get(ModerationStatus.ACTIVE.value, 0),
            jobs_flagged=job_counts.get(ModerationStatus.FLAGGED.value, 0),
            applications_total=applications_total,
            reports_open=reports_open,
            feature_flags_enabled=feature_flags_enabled,
            recent_audit_logs=recent_audit_logs,
            recent_system_logs=recent_system_logs,
        )

    # --- users ---------------------------------------------------------------

    async def list_users(self, filters: UserFilters) -> tuple[list[UserSummary], int]:
        rows, total = await self.repository.list_users(filters)
        summaries = [
            UserSummary(
                user_id=row.user_id,
                status=UserAccountStatus(row.status),
                application_count=row.application_count,
                first_seen_at=row.first_seen_at,
                last_activity_at=row.last_activity_at,
            )
            for row in rows
        ]
        return summaries, total

    async def get_user(self, user_id: uuid.UUID) -> UserDetail:
        summary = await self.repository.user_activity_summary(user_id)
        status_row = await self.repository.user_status_row(user_id)
        status = status_row.status if status_row else UserAccountStatus.ACTIVE
        return UserDetail(
            user_id=user_id,
            status=status,
            application_count=summary["application_count"],
            first_seen_at=summary["first_seen_at"],
            last_activity_at=summary["last_activity_at"],
            reason=status_row.reason if status_row else None,
            changed_at=status_row.changed_at if status_row else None,
            applications_by_status=summary["applications_by_status"],
            auto_apply_task_count=summary["auto_apply_task_count"],
            notification_count=summary["notification_count"],
        )

    async def update_user_status(
        self, actor: AdminActor, user_id: uuid.UUID, request: UserStatusUpdateRequest
    ) -> UserDetail:
        existing = await self.repository.user_status_row(user_id)
        before_status = existing.status if existing else UserAccountStatus.ACTIVE.value
        await self.repository.upsert_user_status(
            user_id=user_id, status=request.status, reason=request.reason, changed_by=actor.id
        )
        await self._audit(
            actor,
            "user.status_update",
            "platform_user",
            user_id,
            before={"status": before_status},
            after={"status": request.status.value, "reason": request.reason},
        )
        return await self.get_user(user_id)

    # --- companies --------------------------------------------------------------

    async def list_companies(self, filters: CompanyFilters) -> tuple[list[CompanySummary], int]:
        rows, total = await self.repository.list_companies(filters)
        summaries = [
            CompanySummary(
                id=company.id,
                name=company.name,
                website=company.website,
                industry=company.industry,
                is_active=company.is_active,
                verification_status=verification.status
                if verification
                else VerificationStatus.UNVERIFIED,
                jobs_count=jobs_count,
                created_at=company.created_at,
            )
            for company, verification, jobs_count in rows
        ]
        return summaries, total

    async def get_company(self, company_id: uuid.UUID) -> CompanyDetail:
        company = await self.repository.get_company(company_id)
        if company is None:
            raise NotFoundError("Company not found.")
        verification = await self.repository.get_or_create_company_verification(company_id)
        jobs_count = await self.repository.company_jobs_count(company_id)
        return CompanyDetail(
            id=company.id,
            name=company.name,
            website=company.website,
            industry=company.industry,
            is_active=company.is_active,
            verification_status=verification.status,
            jobs_count=jobs_count,
            created_at=company.created_at,
            description=company.description,
            logo_url=company.logo_url,
            notes=verification.notes,
            reviewed_by=verification.reviewed_by,
            reviewed_at=verification.reviewed_at,
        )

    async def update_company_verification(
        self, actor: AdminActor, company_id: uuid.UUID, request: CompanyVerificationUpdateRequest
    ) -> CompanyDetail:
        company = await self.repository.get_company(company_id)
        if company is None:
            raise NotFoundError("Company not found.")
        verification = await self.repository.get_or_create_company_verification(company_id)
        before = {"status": verification.status}
        verification.status = request.status
        verification.notes = request.notes
        verification.reviewed_by = actor.id
        verification.reviewed_at = _now()
        await self.repository.session.flush()
        await self._audit(
            actor,
            "company.verification_update",
            "company",
            company_id,
            before=before,
            after={"status": request.status.value, "notes": request.notes},
        )
        return await self.get_company(company_id)

    # --- jobs ---------------------------------------------------------------

    async def list_jobs(self, filters: JobFilters) -> tuple[list[JobSummary], int]:
        rows, total = await self.repository.list_jobs(filters)
        summaries = [
            JobSummary(
                id=job.id,
                title=job.title,
                company_id=job.company_id,
                company_name=company_name,
                status=job.status,
                moderation_status=moderation.status if moderation else ModerationStatus.ACTIVE,
                flag_count=moderation.flag_count if moderation else 0,
                posted_at=job.posted_at,
                created_at=job.created_at,
            )
            for job, company_name, moderation in rows
        ]
        return summaries, total

    async def update_job_moderation(
        self, actor: AdminActor, job_id: uuid.UUID, request: JobModerationUpdateRequest
    ) -> JobSummary:
        job = await self.repository.get_job(job_id)
        if job is None:
            raise NotFoundError("Job not found.")
        moderation = await self.repository.get_or_create_job_moderation(job_id)
        before = {"status": moderation.status}
        moderation.status = request.status
        moderation.reason = request.reason
        moderation.moderated_by = actor.id
        moderation.moderated_at = _now()

        if request.status == ModerationStatus.REMOVED and job.deleted_at is None:
            job.deleted_at = _now()
        elif request.status != ModerationStatus.REMOVED and job.deleted_at is not None:
            job.deleted_at = None

        await self.repository.session.flush()
        await self._audit(
            actor,
            "job.moderation_update",
            "job",
            job_id,
            before=before,
            after={"status": request.status.value, "reason": request.reason},
        )
        company = await self.repository.get_company(job.company_id)
        return JobSummary(
            id=job.id,
            title=job.title,
            company_id=job.company_id,
            company_name=company.name if company else "",
            status=job.status,
            moderation_status=moderation.status,
            flag_count=moderation.flag_count,
            posted_at=job.posted_at,
            created_at=job.created_at,
        )

    # --- applications ---------------------------------------------------------

    async def list_applications(
        self, filters: ApplicationFilters
    ) -> tuple[list[ApplicationSummary], int]:
        rows, total = await self.repository.list_applications(filters)
        summaries = [
            ApplicationSummary(
                id=app.id,
                user_id=app.user_id,
                company_name=app.company_name,
                job_title=app.job_title,
                status=app.status,
                source=app.source,
                applied_at=app.applied_at,
            )
            for app in rows
        ]
        return summaries, total

    async def delete_application(self, actor: AdminActor, application_id: uuid.UUID) -> None:
        application = await self.repository.get_application(application_id)
        if application is None or application.deleted_at is not None:
            raise NotFoundError("Application not found.")
        application.deleted_at = _now()
        await self.repository.session.flush()
        await self._audit(
            actor,
            "application.delete",
            "application",
            application_id,
            before={"user_id": str(application.user_id)},
        )

    # --- reports / content moderation -------------------------------------------

    async def create_report(
        self, reporter_user_id: uuid.UUID, request: ReportCreateRequest
    ) -> ReportResponse:
        report = await self.repository.create_report(
            reporter_user_id=reporter_user_id,
            target_type=request.target_type,
            target_id=request.target_id,
            reason=request.reason,
            status=ReportStatus.OPEN,
        )
        return ReportResponse.model_validate(report)

    async def list_reports(self, filters: ReportFilters) -> tuple[list[ReportResponse], int]:
        rows, total = await self.repository.list_reports(filters)
        return [ReportResponse.model_validate(row) for row in rows], total

    async def update_report(
        self, actor: AdminActor, report_id: uuid.UUID, request: ReportUpdateRequest
    ) -> ReportResponse:
        report = await self.repository.get_report(report_id)
        if report is None:
            raise NotFoundError("Report not found.")
        before = {"status": report.status}
        report.status = request.status
        report.resolution_notes = request.resolution_notes
        if request.status in (ReportStatus.RESOLVED, ReportStatus.DISMISSED):
            report.resolved_by = actor.id
            report.resolved_at = _now()
        await self.repository.session.flush()
        await self._audit(
            actor,
            "report.update",
            "content_report",
            report_id,
            before=before,
            after={"status": request.status.value},
        )
        return ReportResponse.model_validate(report)

    async def get_moderation_queue(self) -> ModerationQueueResponse:
        reports = await self.repository.queue_reports()
        flagged_jobs = await self.repository.queue_flagged_jobs()
        pending_verifications = await self.repository.queue_pending_verifications()

        items = [
            ModerationQueueItem(
                kind="report",
                id=report.id,
                target_type=report.target_type,
                target_id=report.target_id,
                summary=report.reason[:200],
                status=report.status,
                created_at=report.created_at,
            )
            for report in reports
        ] + [
            ModerationQueueItem(
                kind="job",
                id=moderation.id,
                target_type="job",
                target_id=moderation.job_id,
                summary=title,
                status=moderation.status,
                created_at=moderation.updated_at,
            )
            for moderation, title in flagged_jobs
        ] + [
            ModerationQueueItem(
                kind="company",
                id=verification.id,
                target_type="company",
                target_id=verification.company_id,
                summary=name,
                status=verification.status,
                created_at=verification.updated_at,
            )
            for verification, name in pending_verifications
        ]
        items.sort(key=lambda item: item.created_at, reverse=True)
        return ModerationQueueResponse(items=items, total_open=len(items))

    # --- feature flags ------------------------------------------------------------

    async def list_feature_flags(self) -> list[FeatureFlagResponse]:
        flags = await self.repository.list_feature_flags()
        return [FeatureFlagResponse.model_validate(flag) for flag in flags]

    async def create_feature_flag(
        self, actor: AdminActor, request: FeatureFlagCreateRequest
    ) -> FeatureFlagResponse:
        if await self.repository.get_feature_flag_by_key(request.key) is not None:
            raise ConflictError(f"Feature flag '{request.key}' already exists.")
        flag = await self.repository.create_feature_flag(
            key=request.key,
            name=request.name,
            description=request.description,
            is_enabled=request.is_enabled,
            rollout_percentage=request.rollout_percentage,
            updated_by=actor.id,
        )
        await self._audit(
            actor,
            "feature_flag.create",
            "feature_flag",
            flag.id,
            after={"key": flag.key, "is_enabled": flag.is_enabled},
        )
        return FeatureFlagResponse.model_validate(flag)

    async def update_feature_flag(
        self, actor: AdminActor, key: str, request: FeatureFlagUpdateRequest
    ) -> FeatureFlagResponse:
        flag = await self.repository.get_feature_flag_by_key(key)
        if flag is None:
            raise NotFoundError("Feature flag not found.")
        before = {"is_enabled": flag.is_enabled, "rollout_percentage": flag.rollout_percentage}
        for field, value in request.model_dump(exclude_unset=True).items():
            setattr(flag, field, value)
        flag.updated_by = actor.id
        await self.repository.session.flush()
        await self._audit(
            actor,
            "feature_flag.update",
            "feature_flag",
            flag.id,
            before=before,
            after={"is_enabled": flag.is_enabled, "rollout_percentage": flag.rollout_percentage},
        )
        # `updated_at` has `onupdate=func.now()` — after an UPDATE flush its
        # value is server-computed and SQLAlchemy expires the in-memory
        # attribute rather than knowing it, so an unawaited later access
        # would trigger an implicit lazy-load outside any greenlet context
        # (`MissingGreenlet`). Refresh explicitly before Pydantic reads it.
        await self.repository.session.refresh(flag)
        return FeatureFlagResponse.model_validate(flag)

    async def delete_feature_flag(self, actor: AdminActor, key: str) -> None:
        flag = await self.repository.get_feature_flag_by_key(key)
        if flag is None:
            raise NotFoundError("Feature flag not found.")
        flag_id = flag.id
        await self.repository.session.delete(flag)
        await self.repository.session.flush()
        await self._audit(actor, "feature_flag.delete", "feature_flag", flag_id, before={"key": key})

    # --- audit logs / system logs ------------------------------------------------

    async def _build_audit_entries(self, rows) -> list[AuditLogEntry]:
        actor_ids = [row.actor_id for row in rows if row.actor_id is not None]
        email_map = await self.repository.actor_email_map(actor_ids)
        return [
            AuditLogEntry(
                id=row.id,
                actor_id=row.actor_id,
                actor_email=email_map.get(row.actor_id),
                action=row.action,
                target_type=row.target_type,
                target_id=row.target_id,
                before=row.before,
                after=row.after,
                notes=row.notes,
                created_at=row.created_at,
            )
            for row in rows
        ]

    async def list_audit_logs(self, filters: AuditLogFilters) -> tuple[list[AuditLogEntry], int]:
        rows, total = await self.repository.list_audit_logs(filters)
        return await self._build_audit_entries(rows), total

    async def list_system_logs(self, filters: SystemLogFilters):
        return await collect_system_logs(self.repository.session, filters)

    # --- platform analytics --------------------------------------------------------

    async def get_platform_analytics(self, days: int) -> PlatformAnalyticsResponse:
        date_from = _now() - timedelta(days=days)

        def _to_points(rows: list[tuple]) -> list[PlatformTrendPoint]:
            return [PlatformTrendPoint(bucket=bucket.date(), count=count) for bucket, count in rows]

        return PlatformAnalyticsResponse(
            generated_at=_now(),
            days=days,
            active_users_trend=_to_points(await self.repository.active_users_daily_trend(date_from)),
            applications_trend=_to_points(await self.repository.applications_daily_trend(date_from)),
            jobs_posted_trend=_to_points(await self.repository.jobs_posted_daily_trend(date_from)),
            companies_added_trend=_to_points(
                await self.repository.companies_added_daily_trend(date_from)
            ),
            reports_filed_trend=_to_points(await self.repository.reports_filed_daily_trend(date_from)),
            reports_resolved_count=await self.repository.reports_resolved_count(date_from),
            jobs_moderated_count=await self.repository.jobs_moderated_count(date_from),
            companies_reviewed_count=await self.repository.companies_reviewed_count(date_from),
        )


__all__ = ["AdminService"]
