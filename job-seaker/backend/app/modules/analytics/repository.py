"""Analytics module — data access layer.

Pure aggregation queries over other modules' tables (`tracking`,
`resume_parser`, `ats_resume_generator`, `job_ingestion`) — the same
cross-module read convention `ats_resume_generator` and `matching` already
use. This module owns no tables of its own.
"""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import date, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.analytics.schemas import AnalyticsFilters, TrendInterval
from app.modules.ats_resume_generator.models import ATSResume, ATSResumeVersion, ATSVersionStatus
from app.modules.job_ingestion.models import Company, Job, JobStatus, Location, Skill, job_skill
from app.modules.resume_parser.models import (
    Resume,
    ResumeParsedData,
    ResumeVersion,
    ResumeVersionStatus,
)
from app.modules.search.models import SavedJob
from app.modules.tracking.models import Application, ApplicationStatus, ApplicationStatusHistory
from app.repositories.base import BaseRepository

_INTERVIEW_STATUSES = (ApplicationStatus.INTERVIEW, ApplicationStatus.ASSESSMENT)
_OFFER_STATUSES = (ApplicationStatus.OFFER, ApplicationStatus.ACCEPTED)


def _bucket(column, interval: TrendInterval):
    return func.date_trunc(interval.value, column)


class AnalyticsRepository(BaseRepository):
    """Read-only aggregation queries backing the analytics dashboard."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    # --- application trends -------------------------------------------------------

    async def applications_trend(
        self,
        *,
        user_id: uuid.UUID,
        date_from: datetime,
        date_to: datetime,
        interval: TrendInterval,
    ) -> list[tuple[datetime, int]]:
        bucket = _bucket(Application.applied_at, interval).label("bucket")
        stmt = (
            select(bucket, func.count(Application.id))
            .where(
                Application.user_id == user_id,
                Application.deleted_at.is_(None),
                Application.applied_at >= date_from,
                Application.applied_at <= date_to,
            )
            .group_by(bucket)
            .order_by(bucket)
        )
        rows = (await self.session.execute(stmt)).all()
        return [(row[0], row[1]) for row in rows]

    async def status_counts_in_range(
        self, *, user_id: uuid.UUID, date_from: datetime, date_to: datetime
    ) -> dict[ApplicationStatus, int]:
        stmt = (
            select(Application.status, func.count(Application.id))
            .where(
                Application.user_id == user_id,
                Application.deleted_at.is_(None),
                Application.applied_at >= date_from,
                Application.applied_at <= date_to,
            )
            .group_by(Application.status)
        )
        rows = (await self.session.execute(stmt)).all()
        return {row[0]: row[1] for row in rows}

    async def status_transition_trend(
        self,
        *,
        user_id: uuid.UUID,
        to_statuses: Sequence[ApplicationStatus],
        date_from: datetime,
        date_to: datetime,
        interval: TrendInterval,
    ) -> list[tuple[datetime, int]]:
        bucket = _bucket(ApplicationStatusHistory.created_at, interval).label("bucket")
        stmt = (
            select(bucket, func.count(ApplicationStatusHistory.id))
            .select_from(ApplicationStatusHistory)
            .join(Application, Application.id == ApplicationStatusHistory.application_id)
            .where(
                Application.user_id == user_id,
                Application.deleted_at.is_(None),
                ApplicationStatusHistory.to_status.in_(to_statuses),
                ApplicationStatusHistory.created_at >= date_from,
                ApplicationStatusHistory.created_at <= date_to,
            )
            .group_by(bucket)
            .order_by(bucket)
        )
        rows = (await self.session.execute(stmt)).all()
        return [(row[0], row[1]) for row in rows]

    async def acceptance_rate_rows(
        self, *, user_id: uuid.UUID, date_from: datetime, date_to: datetime, interval: TrendInterval
    ) -> list[tuple[datetime, ApplicationStatus, int]]:
        bucket = _bucket(ApplicationStatusHistory.created_at, interval).label("bucket")
        stmt = (
            select(bucket, ApplicationStatusHistory.to_status, func.count(ApplicationStatusHistory.id))
            .select_from(ApplicationStatusHistory)
            .join(Application, Application.id == ApplicationStatusHistory.application_id)
            .where(
                Application.user_id == user_id,
                Application.deleted_at.is_(None),
                ApplicationStatusHistory.to_status.in_(_OFFER_STATUSES),
                ApplicationStatusHistory.created_at >= date_from,
                ApplicationStatusHistory.created_at <= date_to,
            )
            .group_by(bucket, ApplicationStatusHistory.to_status)
            .order_by(bucket)
        )
        rows = (await self.session.execute(stmt)).all()
        return [(row[0], row[1], row[2]) for row in rows]

    # --- resume performance ---------------------------------------------------------

    async def resume_counts(self, *, user_id: uuid.UUID) -> tuple[int, int, int]:
        total_resumes = (
            await self.session.execute(
                select(func.count(Resume.id)).where(
                    Resume.user_id == user_id, Resume.deleted_at.is_(None)
                )
            )
        ).scalar_one()
        total_versions = (
            await self.session.execute(
                select(func.count(ResumeVersion.id))
                .select_from(ResumeVersion)
                .join(Resume, Resume.id == ResumeVersion.resume_id)
                .where(Resume.user_id == user_id, ResumeVersion.deleted_at.is_(None))
            )
        ).scalar_one()
        parsed_versions = (
            await self.session.execute(
                select(func.count(ResumeVersion.id))
                .select_from(ResumeVersion)
                .join(Resume, Resume.id == ResumeVersion.resume_id)
                .where(
                    Resume.user_id == user_id,
                    ResumeVersion.deleted_at.is_(None),
                    ResumeVersion.status == ResumeVersionStatus.PARSED,
                )
            )
        ).scalar_one()
        return total_resumes, total_versions, parsed_versions

    async def resume_versions_trend(
        self, *, user_id: uuid.UUID, date_from: datetime, date_to: datetime, interval: TrendInterval
    ) -> list[tuple[datetime, int]]:
        bucket = _bucket(ResumeVersion.created_at, interval).label("bucket")
        stmt = (
            select(bucket, func.count(ResumeVersion.id))
            .select_from(ResumeVersion)
            .join(Resume, Resume.id == ResumeVersion.resume_id)
            .where(
                Resume.user_id == user_id,
                ResumeVersion.deleted_at.is_(None),
                ResumeVersion.created_at >= date_from,
                ResumeVersion.created_at <= date_to,
            )
            .group_by(bucket)
            .order_by(bucket)
        )
        rows = (await self.session.execute(stmt)).all()
        return [(row[0], row[1]) for row in rows]

    async def average_skills_per_version(self, *, user_id: uuid.UUID) -> float:
        stmt = (
            select(func.avg(func.jsonb_array_length(ResumeParsedData.skills)))
            .select_from(ResumeParsedData)
            .join(ResumeVersion, ResumeVersion.id == ResumeParsedData.resume_version_id)
            .join(Resume, Resume.id == ResumeVersion.resume_id)
            .where(Resume.user_id == user_id, ResumeVersion.deleted_at.is_(None))
        )
        value = (await self.session.execute(stmt)).scalar_one_or_none()
        return round(float(value), 2) if value is not None else 0.0

    async def current_resume_skill_lists(self, *, user_id: uuid.UUID) -> list[list[str]]:
        """Skills from the *current* version of each of the user's resumes.

        Restricted to current versions so a skill isn't over-counted just
        because a resume has many historical versions.
        """
        stmt = (
            select(ResumeParsedData.skills)
            .select_from(ResumeParsedData)
            .join(ResumeVersion, ResumeVersion.id == ResumeParsedData.resume_version_id)
            .join(Resume, Resume.id == ResumeVersion.resume_id)
            .where(
                Resume.user_id == user_id,
                ResumeVersion.deleted_at.is_(None),
                ResumeVersion.is_current.is_(True),
            )
        )
        rows = (await self.session.execute(stmt)).scalars().all()
        return [list(row) for row in rows]

    # --- ATS resume performance -----------------------------------------------------

    async def ats_resume_counts(self, *, user_id: uuid.UUID) -> tuple[int, int, int]:
        total_ats_resumes = (
            await self.session.execute(
                select(func.count(ATSResume.id)).where(
                    ATSResume.user_id == user_id, ATSResume.deleted_at.is_(None)
                )
            )
        ).scalar_one()
        total_versions = (
            await self.session.execute(
                select(func.count(ATSResumeVersion.id))
                .select_from(ATSResumeVersion)
                .join(ATSResume, ATSResume.id == ATSResumeVersion.ats_resume_id)
                .where(ATSResume.user_id == user_id, ATSResumeVersion.deleted_at.is_(None))
            )
        ).scalar_one()
        completed_versions = (
            await self.session.execute(
                select(func.count(ATSResumeVersion.id))
                .select_from(ATSResumeVersion)
                .join(ATSResume, ATSResume.id == ATSResumeVersion.ats_resume_id)
                .where(
                    ATSResume.user_id == user_id,
                    ATSResumeVersion.deleted_at.is_(None),
                    ATSResumeVersion.status == ATSVersionStatus.COMPLETED,
                )
            )
        ).scalar_one()
        return total_ats_resumes, total_versions, completed_versions

    async def ats_score_trend(
        self, *, user_id: uuid.UUID, date_from: datetime, date_to: datetime, interval: TrendInterval
    ) -> list[tuple[datetime, float | None, float | None, int]]:
        bucket = _bucket(ATSResumeVersion.created_at, interval).label("bucket")
        stmt = (
            select(
                bucket,
                func.avg(ATSResumeVersion.ats_score),
                func.avg(ATSResumeVersion.keyword_score),
                func.count(ATSResumeVersion.id),
            )
            .select_from(ATSResumeVersion)
            .join(ATSResume, ATSResume.id == ATSResumeVersion.ats_resume_id)
            .where(
                ATSResume.user_id == user_id,
                ATSResumeVersion.deleted_at.is_(None),
                ATSResumeVersion.status == ATSVersionStatus.COMPLETED,
                ATSResumeVersion.created_at >= date_from,
                ATSResumeVersion.created_at <= date_to,
            )
            .group_by(bucket)
            .order_by(bucket)
        )
        rows = (await self.session.execute(stmt)).all()
        return [
            (row[0], float(row[1]) if row[1] is not None else None,
             float(row[2]) if row[2] is not None else None, row[3])
            for row in rows
        ]

    async def ats_score_values(self, *, user_id: uuid.UUID) -> list[float]:
        stmt = (
            select(ATSResumeVersion.ats_score)
            .select_from(ATSResumeVersion)
            .join(ATSResume, ATSResume.id == ATSResumeVersion.ats_resume_id)
            .where(
                ATSResume.user_id == user_id,
                ATSResumeVersion.deleted_at.is_(None),
                ATSResumeVersion.status == ATSVersionStatus.COMPLETED,
                ATSResumeVersion.ats_score.is_not(None),
            )
        )
        rows = (await self.session.execute(stmt)).scalars().all()
        return [float(v) for v in rows]

    # --- top skills / companies / countries ---------------------------------------

    async def company_outcomes(
        self,
        *,
        user_id: uuid.UUID,
        date_from: datetime,
        date_to: datetime,
        limit: int,
    ) -> list[tuple[str, int, int, int]]:
        # Scoped to this user's own applications via the same join/filter the
        # outer query uses, rather than a bare `SELECT ... FROM
        # application_status_history` — unscoped, these two subqueries would
        # be evaluated over every application's status history on the whole
        # platform on every dashboard load, when only this user's rows can
        # ever match the outer join anyway.
        interviewed = (
            select(ApplicationStatusHistory.application_id)
            .join(Application, Application.id == ApplicationStatusHistory.application_id)
            .where(
                ApplicationStatusHistory.to_status.in_(_INTERVIEW_STATUSES),
                Application.user_id == user_id,
                Application.deleted_at.is_(None),
            )
            .distinct()
            .subquery()
        )
        offered = (
            select(ApplicationStatusHistory.application_id)
            .join(Application, Application.id == ApplicationStatusHistory.application_id)
            .where(
                ApplicationStatusHistory.to_status.in_(_OFFER_STATUSES),
                Application.user_id == user_id,
                Application.deleted_at.is_(None),
            )
            .distinct()
            .subquery()
        )
        stmt = (
            select(
                Application.company_name,
                func.count(Application.id),
                func.count(interviewed.c.application_id),
                func.count(offered.c.application_id),
            )
            .select_from(Application)
            .outerjoin(interviewed, interviewed.c.application_id == Application.id)
            .outerjoin(offered, offered.c.application_id == Application.id)
            .where(
                Application.user_id == user_id,
                Application.deleted_at.is_(None),
                Application.applied_at >= date_from,
                Application.applied_at <= date_to,
            )
            .group_by(Application.company_name)
            .order_by(func.count(Application.id).desc())
            .limit(limit)
        )
        rows = (await self.session.execute(stmt)).all()
        return [(row[0], row[1], row[2], row[3]) for row in rows]

    async def application_locations(
        self, *, user_id: uuid.UUID, date_from: datetime, date_to: datetime
    ) -> list[tuple[str | None, str | None]]:
        stmt = (
            select(Application.location, Location.country)
            .select_from(Application)
            .outerjoin(Job, Job.id == Application.job_id)
            .outerjoin(Location, Location.id == Job.location_id)
            .where(
                Application.user_id == user_id,
                Application.deleted_at.is_(None),
                Application.applied_at >= date_from,
                Application.applied_at <= date_to,
            )
        )
        rows = (await self.session.execute(stmt)).all()
        return [(row[0], row[1]) for row in rows]

    # --- market-wide: hiring trends / skill demand -----------------------------------

    async def job_posting_trend(
        self, *, date_from: datetime, date_to: datetime, interval: TrendInterval
    ) -> list[tuple[datetime, int]]:
        posted_col = func.coalesce(Job.posted_at, Job.created_at)
        bucket = _bucket(posted_col, interval).label("bucket")
        stmt = (
            select(bucket, func.count(Job.id))
            .where(Job.deleted_at.is_(None), posted_col >= date_from, posted_col <= date_to)
            .group_by(bucket)
            .order_by(bucket)
        )
        rows = (await self.session.execute(stmt)).all()
        return [(row[0], row[1]) for row in rows]

    async def top_hiring_companies(
        self, *, date_from: datetime, date_to: datetime, limit: int
    ) -> list[tuple[str, int]]:
        posted_col = func.coalesce(Job.posted_at, Job.created_at)
        stmt = (
            select(Company.name, func.count(Job.id))
            .select_from(Job)
            .join(Company, Company.id == Job.company_id)
            .where(Job.deleted_at.is_(None), posted_col >= date_from, posted_col <= date_to)
            .group_by(Company.name)
            .order_by(func.count(Job.id).desc())
            .limit(limit)
        )
        rows = (await self.session.execute(stmt)).all()
        return [(row[0], row[1]) for row in rows]

    async def skill_demand(self, *, limit: int) -> list[tuple[str, int]]:
        stmt = (
            select(Skill.name, func.count(func.distinct(job_skill.c.job_id)))
            .select_from(Skill)
            .join(job_skill, job_skill.c.skill_id == Skill.id)
            .join(Job, Job.id == job_skill.c.job_id)
            .where(Job.deleted_at.is_(None), Job.status == JobStatus.ACTIVE)
            .group_by(Skill.name)
            .order_by(func.count(func.distinct(job_skill.c.job_id)).desc())
            .limit(limit)
        )
        rows = (await self.session.execute(stmt)).all()
        return [(row[0], row[1]) for row in rows]

    # --- response time ---------------------------------------------------------------

    async def response_time_series(
        self, *, user_id: uuid.UUID, date_from: datetime, date_to: datetime, interval: TrendInterval
    ) -> list[tuple[datetime, float | None, int]]:
        first_change = (
            select(
                ApplicationStatusHistory.application_id,
                func.min(ApplicationStatusHistory.created_at).label("first_changed_at"),
            )
            .group_by(ApplicationStatusHistory.application_id)
            .subquery()
        )
        days_expr = func.extract("epoch", first_change.c.first_changed_at - Application.applied_at) / 86400.0
        bucket = _bucket(Application.applied_at, interval).label("bucket")
        stmt = (
            select(bucket, func.avg(days_expr), func.count(Application.id))
            .select_from(Application)
            .join(first_change, first_change.c.application_id == Application.id)
            .where(
                Application.user_id == user_id,
                Application.deleted_at.is_(None),
                Application.applied_at >= date_from,
                Application.applied_at <= date_to,
            )
            .group_by(bucket)
            .order_by(bucket)
        )
        rows = (await self.session.execute(stmt)).all()
        return [
            (row[0], round(float(row[1]), 2) if row[1] is not None else None, row[2]) for row in rows
        ]

    async def overall_average_response_days(self, *, user_id: uuid.UUID) -> float | None:
        first_change = (
            select(
                ApplicationStatusHistory.application_id,
                func.min(ApplicationStatusHistory.created_at).label("first_changed_at"),
            )
            .group_by(ApplicationStatusHistory.application_id)
            .subquery()
        )
        stmt = (
            select(
                func.avg(
                    func.extract("epoch", first_change.c.first_changed_at - Application.applied_at)
                )
            )
            .select_from(Application)
            .join(first_change, first_change.c.application_id == Application.id)
            .where(Application.user_id == user_id, Application.deleted_at.is_(None))
        )
        seconds = (await self.session.execute(stmt)).scalar_one_or_none()
        return round(float(seconds) / 86400, 2) if seconds is not None else None

    # --- heatmap -----------------------------------------------------------------------

    async def application_counts_by_date(
        self, *, user_id: uuid.UUID, date_from: datetime, date_to: datetime
    ) -> list[tuple[date, int]]:
        day = func.date(Application.applied_at).label("day")
        stmt = (
            select(day, func.count(Application.id))
            .where(
                Application.user_id == user_id,
                Application.deleted_at.is_(None),
                Application.applied_at >= date_from,
                Application.applied_at <= date_to,
            )
            .group_by(day)
            .order_by(day)
        )
        rows = (await self.session.execute(stmt)).all()
        return [(row[0], row[1]) for row in rows]

    # --- Job Search integration (Section 8) ------------------------------------------

    async def count_saved_jobs(self, *, user_id: uuid.UUID) -> int:
        stmt = select(func.count(SavedJob.id)).where(SavedJob.user_id == user_id)
        result = await self.session.execute(stmt)
        return result.scalar_one()

    # --- export ------------------------------------------------------------------------

    async def list_for_export(
        self, *, user_id: uuid.UUID, filters: AnalyticsFilters
    ) -> Sequence[Application]:
        stmt = select(Application).where(
            Application.user_id == user_id, Application.deleted_at.is_(None)
        )
        if filters.date_from is not None:
            stmt = stmt.where(Application.applied_at >= filters.date_from)
        if filters.date_to is not None:
            stmt = stmt.where(Application.applied_at <= filters.date_to)
        if filters.status:
            stmt = stmt.where(Application.status.in_(filters.status))
        if filters.company_name:
            stmt = stmt.where(Application.company_name.ilike(f"%{filters.company_name}%"))
        stmt = stmt.order_by(Application.applied_at.desc())
        result = await self.session.execute(stmt)
        return result.scalars().all()


__all__ = ["AnalyticsRepository"]
