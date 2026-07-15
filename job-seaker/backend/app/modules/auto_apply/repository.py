"""Auto Apply module — data access layer.

Reads `job_ingestion` (target job + company intelligence), `ats_resume_generator`,
and `resume_parser` (candidate resumes for `resume_selection.py`) directly —
the same cross-module read convention `ats_resume_generator` already uses.
"""
from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.modules.ats_resume_generator.models import ATSResume, ATSResumeVersion
from app.modules.auto_apply.models import (
    ApplicantProfile,
    AutoApplyAttempt,
    AutoApplyLogEntry,
    AutoApplyScreenshot,
    AutoApplyTask,
    AutoApplyTaskStatus,
)
from app.modules.auto_apply.schemas import AutoApplyTaskFilters
from app.modules.job_ingestion.models import Company, CompanyIntelligence, Job
from app.modules.resume_parser.models import Resume
from app.repositories.base import BaseRepository

_ACTIVE_TASK_STATUSES = (
    AutoApplyTaskStatus.PENDING,
    AutoApplyTaskStatus.QUEUED,
    AutoApplyTaskStatus.RUNNING,
    AutoApplyTaskStatus.RETRYING,
    AutoApplyTaskStatus.REQUIRES_REVIEW,
)
_ATS_RESUME_EAGER_LOAD = (selectinload(ATSResume.versions).selectinload(ATSResumeVersion.files),)
_RESUME_EAGER_LOAD = (selectinload(Resume.versions),)
_ATTEMPT_EAGER_LOAD = (
    selectinload(AutoApplyAttempt.screenshots),
    selectinload(AutoApplyAttempt.logs),
)


class ApplicantProfileRepository(BaseRepository[ApplicantProfile]):
    model = ApplicantProfile

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_by_user(self, *, user_id: uuid.UUID) -> ApplicantProfile | None:
        stmt = select(ApplicantProfile).where(ApplicantProfile.user_id == user_id)
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def get_or_create(self, *, user_id: uuid.UUID) -> ApplicantProfile:
        existing = await self.get_by_user(user_id=user_id)
        if existing is not None:
            return existing
        return await self.create(user_id=user_id)


class AutoApplyTaskRepository(BaseRepository[AutoApplyTask]):
    model = AutoApplyTask

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    def _base_stmt(self) -> Select:
        return select(AutoApplyTask).options(selectinload(AutoApplyTask.attempts))

    async def get_owned(self, *, task_id: uuid.UUID, user_id: uuid.UUID) -> AutoApplyTask | None:
        stmt = self._base_stmt().where(
            AutoApplyTask.id == task_id,
            AutoApplyTask.user_id == user_id,
            AutoApplyTask.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().one_or_none()

    async def get_for_worker(self, *, task_id: uuid.UUID) -> AutoApplyTask | None:
        """Fetch by id alone (no owner check) — used by the Celery task path."""
        stmt = self._base_stmt().where(AutoApplyTask.id == task_id, AutoApplyTask.deleted_at.is_(None))
        result = await self.session.execute(stmt)
        return result.scalars().unique().one_or_none()

    async def get_active_for_job(self, *, user_id: uuid.UUID, job_id: uuid.UUID) -> AutoApplyTask | None:
        stmt = select(AutoApplyTask).where(
            AutoApplyTask.user_id == user_id,
            AutoApplyTask.job_id == job_id,
            AutoApplyTask.deleted_at.is_(None),
            AutoApplyTask.status.in_(_ACTIVE_TASK_STATUSES),
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    def _apply_filters(self, stmt: Select, *, user_id: uuid.UUID, filters: AutoApplyTaskFilters) -> Select:
        stmt = stmt.where(AutoApplyTask.user_id == user_id, AutoApplyTask.deleted_at.is_(None))
        if filters.status:
            stmt = stmt.where(AutoApplyTask.status.in_(filters.status))
        if filters.job_id:
            stmt = stmt.where(AutoApplyTask.job_id == filters.job_id)
        return stmt

    async def list_filtered(
        self, *, user_id: uuid.UUID, filters: AutoApplyTaskFilters, offset: int, limit: int
    ) -> Sequence[AutoApplyTask]:
        stmt = self._apply_filters(self._base_stmt(), user_id=user_id, filters=filters)
        stmt = stmt.order_by(AutoApplyTask.created_at.desc()).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().unique().all()

    async def count_filtered(self, *, user_id: uuid.UUID, filters: AutoApplyTaskFilters) -> int:
        stmt = self._apply_filters(
            select(func.count(AutoApplyTask.id)), user_id=user_id, filters=filters
        )
        result = await self.session.execute(stmt)
        return result.scalar_one()

    # --- cross-module reads ------------------------------------------------------

    async def get_job_with_company(self, *, job_id: uuid.UUID) -> Job | None:
        stmt = (
            select(Job)
            .where(Job.id == job_id, Job.deleted_at.is_(None))
            .options(
                selectinload(Job.company).selectinload(Company.intelligence),
            )
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().one_or_none()

    async def list_ats_resumes_for_user(self, *, user_id: uuid.UUID) -> Sequence[ATSResume]:
        stmt = (
            select(ATSResume)
            .where(ATSResume.user_id == user_id, ATSResume.deleted_at.is_(None))
            .options(*_ATS_RESUME_EAGER_LOAD)
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().all()

    async def list_resumes_for_user(self, *, user_id: uuid.UUID) -> Sequence[Resume]:
        stmt = (
            select(Resume)
            .where(Resume.user_id == user_id, Resume.deleted_at.is_(None))
            .options(*_RESUME_EAGER_LOAD)
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().all()

    async def get_company_intelligence(self, *, company_id: uuid.UUID) -> CompanyIntelligence | None:
        stmt = select(CompanyIntelligence).where(CompanyIntelligence.company_id == company_id)
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()


class AutoApplyAttemptRepository(BaseRepository[AutoApplyAttempt]):
    model = AutoApplyAttempt

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_for_task(self, *, attempt_id: uuid.UUID, task_id: uuid.UUID) -> AutoApplyAttempt | None:
        stmt = (
            select(AutoApplyAttempt)
            .where(AutoApplyAttempt.id == attempt_id, AutoApplyAttempt.task_id == task_id)
            .options(*_ATTEMPT_EAGER_LOAD)
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().one_or_none()

    async def list_for_task(self, *, task_id: uuid.UUID) -> Sequence[AutoApplyAttempt]:
        stmt = (
            select(AutoApplyAttempt)
            .where(AutoApplyAttempt.task_id == task_id)
            .options(*_ATTEMPT_EAGER_LOAD)
            .order_by(AutoApplyAttempt.attempt_number.desc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().all()

    async def add_screenshot(
        self, *, attempt_id: uuid.UUID, stage: str, s3_bucket: str, s3_key: str
    ) -> AutoApplyScreenshot:
        screenshot = AutoApplyScreenshot(
            attempt_id=attempt_id, stage=stage, s3_bucket=s3_bucket, s3_key=s3_key
        )
        self.session.add(screenshot)
        await self.session.flush()
        return screenshot

    async def add_log_entry(
        self, *, attempt_id: uuid.UUID, level: str, message: str, context: dict
    ) -> AutoApplyLogEntry:
        entry = AutoApplyLogEntry(attempt_id=attempt_id, level=level, message=message, context=context)
        self.session.add(entry)
        await self.session.flush()
        return entry

    async def get_screenshot(
        self, *, screenshot_id: uuid.UUID, attempt_id: uuid.UUID
    ) -> AutoApplyScreenshot | None:
        stmt = select(AutoApplyScreenshot).where(
            AutoApplyScreenshot.id == screenshot_id, AutoApplyScreenshot.attempt_id == attempt_id
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()


__all__ = ["ApplicantProfileRepository", "AutoApplyTaskRepository", "AutoApplyAttemptRepository"]
