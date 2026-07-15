"""Admin module — data access layer.

Owns the 7 admin tables (`models.py`) and reads across module boundaries
into `job_ingestion` (`Company`, `Job`) and `tracking` (`Application`) —
plus lighter read-only touches on `auto_apply`/`notification` for user
activity stats — the same cross-module read convention `analytics.repository`
already uses.
"""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import String, cast, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.admin.models import (
    AdminActor,
    AuditLog,
    CompanyVerification,
    ContentReport,
    FeatureFlag,
    JobModeration,
    PlatformUserStatus,
    UserAccountStatus,
)
from app.modules.admin.schemas import (
    ApplicationFilters,
    AuditLogFilters,
    CompanyFilters,
    JobFilters,
    ReportFilters,
    UserFilters,
)
from app.modules.auto_apply.models import AutoApplyTask
from app.modules.job_ingestion.models import Company, Job
from app.modules.notification.models import Notification
from app.modules.tracking.models import Application
from app.repositories.base import BaseRepository

_ACTIVE = UserAccountStatus.ACTIVE.value


class AdminRepository(BaseRepository):
    """Data access for the Admin module."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    # --- actors ---------------------------------------------------------------

    async def get_actor_by_user_id(self, user_id: uuid.UUID) -> AdminActor | None:
        stmt = select(AdminActor).where(AdminActor.user_id == user_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_actor(self, actor_id: uuid.UUID) -> AdminActor | None:
        return await self.session.get(AdminActor, actor_id)

    async def count_actors(self) -> int:
        stmt = select(func.count()).select_from(AdminActor)
        return (await self.session.execute(stmt)).scalar_one()

    _BOOTSTRAP_LOCK_KEY = 727_100_001

    async def acquire_bootstrap_lock(self) -> None:
        """Transaction-scoped advisory lock serializing concurrent bootstrap
        attempts, so the count-then-create in `AdminService.bootstrap_actor`
        can't race two callers into both becoming SUPER_ADMIN. Released
        automatically on commit/rollback of the request's transaction
        (`app.db.session.get_db`) — no manual unlock needed."""
        await self.session.execute(
            text("SELECT pg_advisory_xact_lock(:key)"), {"key": self._BOOTSTRAP_LOCK_KEY}
        )

    async def list_actors(self, *, offset: int, limit: int) -> Sequence[AdminActor]:
        stmt = select(AdminActor).order_by(AdminActor.created_at.desc()).offset(offset).limit(limit)
        return (await self.session.execute(stmt)).scalars().all()

    async def create_actor(self, **kwargs) -> AdminActor:
        actor = AdminActor(**kwargs)
        self.session.add(actor)
        await self.session.flush()
        return actor

    async def update_actor(self, actor: AdminActor, **kwargs) -> AdminActor:
        for field, value in kwargs.items():
            setattr(actor, field, value)
        await self.session.flush()
        return actor

    async def actor_email_map(self, actor_ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, str]:
        if not actor_ids:
            return {}
        stmt = select(AdminActor.id, AdminActor.email).where(AdminActor.id.in_(actor_ids))
        rows = (await self.session.execute(stmt)).all()
        return {row[0]: row[1] for row in rows}

    # --- users (aggregated over `tracking.Application`, overlaid with status) -

    def _user_status_column(self):
        return func.coalesce(PlatformUserStatus.status, _ACTIVE)

    async def list_users(self, filters: UserFilters) -> tuple[list[tuple], int]:
        status_col = self._user_status_column().label("status")
        base = (
            select(
                Application.user_id,
                func.count(Application.id).label("application_count"),
                func.min(Application.applied_at).label("first_seen_at"),
                func.max(Application.applied_at).label("last_activity_at"),
                status_col,
            )
            .select_from(Application)
            .outerjoin(PlatformUserStatus, PlatformUserStatus.user_id == Application.user_id)
            .where(Application.deleted_at.is_(None))
            .group_by(Application.user_id, PlatformUserStatus.status)
        )
        if filters.search:
            base = base.where(cast(Application.user_id, String).ilike(f"%{filters.search}%"))
        if filters.status is not None:
            base = base.having(status_col == filters.status.value)

        count_stmt = select(func.count()).select_from(base.subquery())
        total = (await self.session.execute(count_stmt)).scalar_one()

        page_stmt = (
            base.order_by(func.max(Application.applied_at).desc())
            .offset(filters.offset)
            .limit(filters.page_size)
        )
        rows = (await self.session.execute(page_stmt)).all()
        return rows, total

    async def user_status_row(self, user_id: uuid.UUID) -> PlatformUserStatus | None:
        stmt = select(PlatformUserStatus).where(PlatformUserStatus.user_id == user_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def user_activity_summary(self, user_id: uuid.UUID) -> dict:
        app_stmt = (
            select(Application.status, func.count(Application.id))
            .where(Application.user_id == user_id, Application.deleted_at.is_(None))
            .group_by(Application.status)
        )
        app_rows = (await self.session.execute(app_stmt)).all()
        # `row[0]` is the raw column value SQLAlchemy returns for a
        # `String`-backed enum column — always a plain `str` at runtime (the
        # `Mapped[SomeEnum]` annotation is a static-typing hint only), never
        # an actual enum instance, so no `.value` here.
        applications_by_status = {row[0]: row[1] for row in app_rows}

        seen_stmt = select(
            func.min(Application.applied_at), func.max(Application.applied_at)
        ).where(Application.user_id == user_id, Application.deleted_at.is_(None))
        first_seen_at, last_activity_at = (await self.session.execute(seen_stmt)).one()

        auto_apply_stmt = select(func.count()).select_from(AutoApplyTask).where(
            AutoApplyTask.user_id == user_id, AutoApplyTask.deleted_at.is_(None)
        )
        auto_apply_task_count = (await self.session.execute(auto_apply_stmt)).scalar_one()

        notification_stmt = select(func.count()).select_from(Notification).where(
            Notification.user_id == user_id
        )
        notification_count = (await self.session.execute(notification_stmt)).scalar_one()

        return {
            "applications_by_status": applications_by_status,
            "application_count": sum(applications_by_status.values()),
            "first_seen_at": first_seen_at,
            "last_activity_at": last_activity_at,
            "auto_apply_task_count": auto_apply_task_count,
            "notification_count": notification_count,
        }

    async def upsert_user_status(
        self, *, user_id: uuid.UUID, status: UserAccountStatus, reason: str | None, changed_by: uuid.UUID
    ) -> PlatformUserStatus:
        row = await self.user_status_row(user_id)
        now = datetime.utcnow()
        if row is None:
            row = PlatformUserStatus(
                user_id=user_id, status=status, reason=reason, changed_by=changed_by, changed_at=now
            )
            self.session.add(row)
        else:
            row.status = status
            row.reason = reason
            row.changed_by = changed_by
            row.changed_at = now
        await self.session.flush()
        return row

    async def user_counts_by_status(self) -> dict[str, int]:
        """Counts overridden users by status; callers add the ACTIVE default separately."""
        stmt = select(PlatformUserStatus.status, func.count()).group_by(PlatformUserStatus.status)
        rows = (await self.session.execute(stmt)).all()
        return {row[0]: row[1] for row in rows}

    async def distinct_user_count(self) -> int:
        stmt = select(func.count(func.distinct(Application.user_id))).where(
            Application.deleted_at.is_(None)
        )
        return (await self.session.execute(stmt)).scalar_one()

    # --- companies --------------------------------------------------------------

    async def list_companies(self, filters: CompanyFilters) -> tuple[Sequence[tuple], int]:
        jobs_count = (
            select(func.count(Job.id))
            .where(Job.company_id == Company.id, Job.deleted_at.is_(None))
            .correlate(Company)
            .scalar_subquery()
        )
        base = (
            select(Company, CompanyVerification, jobs_count.label("jobs_count"))
            .select_from(Company)
            .outerjoin(CompanyVerification, CompanyVerification.company_id == Company.id)
            .where(Company.deleted_at.is_(None))
        )
        if filters.search:
            base = base.where(Company.name.ilike(f"%{filters.search}%"))
        if filters.is_active is not None:
            base = base.where(Company.is_active == filters.is_active)
        if filters.verification_status is not None:
            status_col = func.coalesce(CompanyVerification.status, "unverified")
            base = base.where(status_col == filters.verification_status.value)

        count_stmt = select(func.count()).select_from(base.with_only_columns(Company.id).subquery())
        total = (await self.session.execute(count_stmt)).scalar_one()

        page_stmt = base.order_by(Company.created_at.desc()).offset(filters.offset).limit(filters.page_size)
        rows = (await self.session.execute(page_stmt)).all()
        return rows, total

    async def get_company(self, company_id: uuid.UUID) -> Company | None:
        return await self.session.get(Company, company_id)

    async def get_or_create_company_verification(self, company_id: uuid.UUID) -> CompanyVerification:
        stmt = select(CompanyVerification).where(CompanyVerification.company_id == company_id)
        row = (await self.session.execute(stmt)).scalar_one_or_none()
        if row is None:
            row = CompanyVerification(company_id=company_id)
            self.session.add(row)
            await self.session.flush()
        return row

    async def company_jobs_count(self, company_id: uuid.UUID) -> int:
        stmt = select(func.count()).select_from(Job).where(
            Job.company_id == company_id, Job.deleted_at.is_(None)
        )
        return (await self.session.execute(stmt)).scalar_one()

    async def company_verification_counts(self) -> dict[str, int]:
        total_stmt = select(func.count()).select_from(Company).where(Company.deleted_at.is_(None))
        total = (await self.session.execute(total_stmt)).scalar_one()
        status_stmt = select(CompanyVerification.status, func.count()).group_by(
            CompanyVerification.status
        )
        rows = (await self.session.execute(status_stmt)).all()
        return {"total": total, **{row[0]: row[1] for row in rows}}

    # --- jobs ---------------------------------------------------------------

    async def list_jobs(self, filters: JobFilters) -> tuple[Sequence[tuple], int]:
        base = (
            select(Job, Company.name, JobModeration)
            .select_from(Job)
            .join(Company, Company.id == Job.company_id)
            .outerjoin(JobModeration, JobModeration.job_id == Job.id)
            .where(Job.deleted_at.is_(None))
        )
        if filters.search:
            base = base.where(Job.title.ilike(f"%{filters.search}%"))
        if filters.company_id is not None:
            base = base.where(Job.company_id == filters.company_id)
        if filters.job_status is not None:
            base = base.where(Job.status == filters.job_status)
        if filters.moderation_status is not None:
            status_col = func.coalesce(JobModeration.status, "active")
            base = base.where(status_col == filters.moderation_status.value)

        count_stmt = select(func.count()).select_from(base.with_only_columns(Job.id).subquery())
        total = (await self.session.execute(count_stmt)).scalar_one()

        page_stmt = base.order_by(Job.created_at.desc()).offset(filters.offset).limit(filters.page_size)
        rows = (await self.session.execute(page_stmt)).all()
        return rows, total

    async def get_job(self, job_id: uuid.UUID) -> Job | None:
        return await self.session.get(Job, job_id)

    async def get_or_create_job_moderation(self, job_id: uuid.UUID) -> JobModeration:
        stmt = select(JobModeration).where(JobModeration.job_id == job_id)
        row = (await self.session.execute(stmt)).scalar_one_or_none()
        if row is None:
            row = JobModeration(job_id=job_id)
            self.session.add(row)
            await self.session.flush()
        return row

    async def job_moderation_counts(self) -> dict[str, int]:
        total_stmt = select(func.count()).select_from(Job).where(Job.deleted_at.is_(None))
        total = (await self.session.execute(total_stmt)).scalar_one()
        status_stmt = select(JobModeration.status, func.count()).group_by(JobModeration.status)
        rows = (await self.session.execute(status_stmt)).all()
        return {"total": total, **{row[0]: row[1] for row in rows}}

    # --- applications ---------------------------------------------------------

    async def list_applications(self, filters: ApplicationFilters) -> tuple[Sequence[Application], int]:
        base = select(Application).where(Application.deleted_at.is_(None))
        if filters.user_id is not None:
            base = base.where(Application.user_id == filters.user_id)
        if filters.status is not None:
            base = base.where(Application.status == filters.status)
        if filters.search:
            like = f"%{filters.search}%"
            base = base.where(
                (Application.company_name.ilike(like)) | (Application.job_title.ilike(like))
            )

        count_stmt = select(func.count()).select_from(base.with_only_columns(Application.id).subquery())
        total = (await self.session.execute(count_stmt)).scalar_one()

        page_stmt = (
            base.order_by(Application.applied_at.desc()).offset(filters.offset).limit(filters.page_size)
        )
        rows = (await self.session.execute(page_stmt)).scalars().all()
        return rows, total

    async def get_application(self, application_id: uuid.UUID) -> Application | None:
        return await self.session.get(Application, application_id)

    async def applications_total(self) -> int:
        stmt = select(func.count()).select_from(Application).where(Application.deleted_at.is_(None))
        return (await self.session.execute(stmt)).scalar_one()

    # --- reports --------------------------------------------------------------

    async def list_reports(self, filters: ReportFilters) -> tuple[Sequence[ContentReport], int]:
        base = select(ContentReport)
        if filters.status is not None:
            base = base.where(ContentReport.status == filters.status)
        if filters.target_type is not None:
            base = base.where(ContentReport.target_type == filters.target_type)

        count_stmt = select(func.count()).select_from(base.with_only_columns(ContentReport.id).subquery())
        total = (await self.session.execute(count_stmt)).scalar_one()

        page_stmt = (
            base.order_by(ContentReport.created_at.desc()).offset(filters.offset).limit(filters.page_size)
        )
        rows = (await self.session.execute(page_stmt)).scalars().all()
        return rows, total

    async def create_report(self, **kwargs) -> ContentReport:
        report = ContentReport(**kwargs)
        self.session.add(report)
        await self.session.flush()
        return report

    async def get_report(self, report_id: uuid.UUID) -> ContentReport | None:
        return await self.session.get(ContentReport, report_id)

    async def open_reports_count(self) -> int:
        stmt = select(func.count()).select_from(ContentReport).where(ContentReport.status == "open")
        return (await self.session.execute(stmt)).scalar_one()

    # --- feature flags ----------------------------------------------------------

    async def list_feature_flags(self) -> Sequence[FeatureFlag]:
        stmt = select(FeatureFlag).order_by(FeatureFlag.key)
        return (await self.session.execute(stmt)).scalars().all()

    async def get_feature_flag_by_key(self, key: str) -> FeatureFlag | None:
        stmt = select(FeatureFlag).where(FeatureFlag.key == key)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def create_feature_flag(self, **kwargs) -> FeatureFlag:
        flag = FeatureFlag(**kwargs)
        self.session.add(flag)
        await self.session.flush()
        return flag

    async def enabled_feature_flags_count(self) -> int:
        stmt = select(func.count()).select_from(FeatureFlag).where(FeatureFlag.is_enabled.is_(True))
        return (await self.session.execute(stmt)).scalar_one()

    # --- audit logs -------------------------------------------------------------

    async def create_audit_log(self, **kwargs) -> AuditLog:
        entry = AuditLog(**kwargs)
        self.session.add(entry)
        await self.session.flush()
        return entry

    async def list_audit_logs(self, filters: AuditLogFilters) -> tuple[Sequence[AuditLog], int]:
        base = select(AuditLog)
        if filters.actor_id is not None:
            base = base.where(AuditLog.actor_id == filters.actor_id)
        if filters.action is not None:
            base = base.where(AuditLog.action == filters.action)
        if filters.target_type is not None:
            base = base.where(AuditLog.target_type == filters.target_type)
        if filters.date_from is not None:
            base = base.where(AuditLog.created_at >= filters.date_from)
        if filters.date_to is not None:
            base = base.where(AuditLog.created_at <= filters.date_to)

        count_stmt = select(func.count()).select_from(base.with_only_columns(AuditLog.id).subquery())
        total = (await self.session.execute(count_stmt)).scalar_one()

        page_stmt = (
            base.order_by(AuditLog.created_at.desc()).offset(filters.offset).limit(filters.page_size)
        )
        rows = (await self.session.execute(page_stmt)).scalars().all()
        return rows, total

    async def recent_audit_logs(self, limit: int) -> Sequence[AuditLog]:
        stmt = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
        return (await self.session.execute(stmt)).scalars().all()

    # --- platform analytics (daily trends over the last N days) ----------------

    async def _daily_trend(self, column, date_from: datetime, extra_where=None) -> list[tuple]:
        bucket = func.date_trunc("day", column).label("bucket")
        stmt = select(bucket, func.count()).where(column >= date_from).group_by(bucket).order_by(bucket)
        if extra_where is not None:
            stmt = stmt.where(extra_where)
        rows = (await self.session.execute(stmt)).all()
        return [(row[0], row[1]) for row in rows]

    async def applications_daily_trend(self, date_from: datetime) -> list[tuple]:
        return await self._daily_trend(
            Application.applied_at, date_from, Application.deleted_at.is_(None)
        )

    async def jobs_posted_daily_trend(self, date_from: datetime) -> list[tuple]:
        return await self._daily_trend(Job.created_at, date_from, Job.deleted_at.is_(None))

    async def companies_added_daily_trend(self, date_from: datetime) -> list[tuple]:
        return await self._daily_trend(Company.created_at, date_from, Company.deleted_at.is_(None))

    async def reports_filed_daily_trend(self, date_from: datetime) -> list[tuple]:
        return await self._daily_trend(ContentReport.created_at, date_from)

    async def active_users_daily_trend(self, date_from: datetime) -> list[tuple]:
        bucket = func.date_trunc("day", Application.applied_at).label("bucket")
        stmt = (
            select(bucket, func.count(func.distinct(Application.user_id)))
            .where(Application.applied_at >= date_from, Application.deleted_at.is_(None))
            .group_by(bucket)
            .order_by(bucket)
        )
        rows = (await self.session.execute(stmt)).all()
        return [(row[0], row[1]) for row in rows]

    async def reports_resolved_count(self, date_from: datetime) -> int:
        stmt = select(func.count()).select_from(ContentReport).where(
            ContentReport.resolved_at.is_not(None), ContentReport.resolved_at >= date_from
        )
        return (await self.session.execute(stmt)).scalar_one()

    async def jobs_moderated_count(self, date_from: datetime) -> int:
        stmt = select(func.count()).select_from(JobModeration).where(
            JobModeration.moderated_at.is_not(None), JobModeration.moderated_at >= date_from
        )
        return (await self.session.execute(stmt)).scalar_one()

    async def companies_reviewed_count(self, date_from: datetime) -> int:
        stmt = select(func.count()).select_from(CompanyVerification).where(
            CompanyVerification.reviewed_at.is_not(None), CompanyVerification.reviewed_at >= date_from
        )
        return (await self.session.execute(stmt)).scalar_one()

    # --- moderation queue --------------------------------------------------------

    async def queue_reports(self, limit: int = 100) -> Sequence[ContentReport]:
        stmt = (
            select(ContentReport)
            .where(ContentReport.status.in_(["open", "reviewing"]))
            .order_by(ContentReport.created_at.desc())
            .limit(limit)
        )
        return (await self.session.execute(stmt)).scalars().all()

    async def queue_flagged_jobs(self, limit: int = 100) -> Sequence[tuple]:
        stmt = (
            select(JobModeration, Job.title)
            .join(Job, Job.id == JobModeration.job_id)
            .where(JobModeration.status.in_(["flagged", "pending_review"]))
            .order_by(JobModeration.updated_at.desc())
            .limit(limit)
        )
        return (await self.session.execute(stmt)).all()

    async def queue_pending_verifications(self, limit: int = 100) -> Sequence[tuple]:
        stmt = (
            select(CompanyVerification, Company.name)
            .join(Company, Company.id == CompanyVerification.company_id)
            .where(CompanyVerification.status == "pending")
            .order_by(CompanyVerification.updated_at.desc())
            .limit(limit)
        )
        return (await self.session.execute(stmt)).all()


__all__ = ["AdminRepository"]
