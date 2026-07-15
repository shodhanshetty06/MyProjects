"""ATS Resume Generator module — data access layer.

Reads `resume_parser` (source resume content) and `job_ingestion` (optional
target job) models directly — the same cross-module read convention
`matching` already uses — in addition to owning this module's own three
tables.
"""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.modules.ats_resume_generator.models import ATSResume, ATSResumeFile, ATSResumeVersion
from app.modules.job_ingestion.models import Job
from app.modules.resume_parser.models import Resume, ResumeVersion
from app.repositories.base import BaseRepository

_VERSION_EAGER_LOAD = (selectinload(ATSResumeVersion.files),)


class ATSResumeRepository(BaseRepository[ATSResume]):
    """Data access for the logical `ATSResume` entity."""

    model = ATSResume

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def list_by_user(
        self, *, user_id: uuid.UUID, offset: int = 0, limit: int = 20
    ) -> Sequence[ATSResume]:
        stmt = (
            select(ATSResume)
            .where(ATSResume.user_id == user_id, ATSResume.deleted_at.is_(None))
            .options(selectinload(ATSResume.versions).options(*_VERSION_EAGER_LOAD))
            .order_by(ATSResume.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().all()

    async def count_by_user(self, *, user_id: uuid.UUID) -> int:
        stmt = select(func.count()).select_from(ATSResume).where(
            ATSResume.user_id == user_id, ATSResume.deleted_at.is_(None)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def get_owned(self, *, ats_resume_id: uuid.UUID, user_id: uuid.UUID) -> ATSResume | None:
        stmt = (
            select(ATSResume)
            .where(
                ATSResume.id == ats_resume_id,
                ATSResume.user_id == user_id,
                ATSResume.deleted_at.is_(None),
            )
            .options(selectinload(ATSResume.versions).options(*_VERSION_EAGER_LOAD))
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().one_or_none()


class ATSResumeVersionRepository(BaseRepository[ATSResumeVersion]):
    """Data access for `ATSResumeVersion`, including current-version bookkeeping."""

    model = ATSResumeVersion

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def list_for_resume(self, *, ats_resume_id: uuid.UUID) -> Sequence[ATSResumeVersion]:
        stmt = (
            select(ATSResumeVersion)
            .where(
                ATSResumeVersion.ats_resume_id == ats_resume_id,
                ATSResumeVersion.deleted_at.is_(None),
            )
            .options(*_VERSION_EAGER_LOAD)
            .order_by(ATSResumeVersion.version_number.desc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_for_resume(
        self, *, ats_resume_id: uuid.UUID, version_id: uuid.UUID
    ) -> ATSResumeVersion | None:
        stmt = (
            select(ATSResumeVersion)
            .where(
                ATSResumeVersion.id == version_id,
                ATSResumeVersion.ats_resume_id == ats_resume_id,
                ATSResumeVersion.deleted_at.is_(None),
            )
            .options(*_VERSION_EAGER_LOAD)
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def get_by_id(self, *, version_id: uuid.UUID) -> ATSResumeVersion | None:
        """Fetch a version by id alone (no owner check) — used by the Celery
        task path, which only receives the version id.
        """
        stmt = (
            select(ATSResumeVersion)
            .where(ATSResumeVersion.id == version_id, ATSResumeVersion.deleted_at.is_(None))
            .options(*_VERSION_EAGER_LOAD, selectinload(ATSResumeVersion.ats_resume))
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def next_version_number(self, *, ats_resume_id: uuid.UUID) -> int:
        stmt = select(func.max(ATSResumeVersion.version_number)).where(
            ATSResumeVersion.ats_resume_id == ats_resume_id
        )
        result = await self.session.execute(stmt)
        current_max = result.scalar_one_or_none()
        return (current_max or 0) + 1

    async def set_current(self, *, ats_resume_id: uuid.UUID, version_id: uuid.UUID) -> None:
        """Atomically make `version_id` the only current version of `ats_resume_id`."""
        await self.session.execute(
            update(ATSResumeVersion)
            .where(
                ATSResumeVersion.ats_resume_id == ats_resume_id,
                ATSResumeVersion.id != version_id,
            )
            .values(is_current=False)
        )
        await self.session.execute(
            update(ATSResumeVersion).where(ATSResumeVersion.id == version_id).values(is_current=True)
        )
        await self.session.flush()

    async def add_file(
        self,
        *,
        ats_resume_version_id: uuid.UUID,
        format: Any,
        s3_bucket: str,
        s3_key: str,
        file_size_bytes: int,
    ) -> ATSResumeFile:
        """Insert-or-replace the file row for `(version, format)` — a
        regenerated version's files supersede any prior artifact rather than
        accumulating duplicates.
        """
        stmt = select(ATSResumeFile).where(
            ATSResumeFile.ats_resume_version_id == ats_resume_version_id,
            ATSResumeFile.format == format,
        )
        existing = (await self.session.execute(stmt)).scalars().one_or_none()
        if existing is not None:
            existing.s3_bucket = s3_bucket
            existing.s3_key = s3_key
            existing.file_size_bytes = file_size_bytes
            await self.session.flush()
            return existing

        file_row = ATSResumeFile(
            ats_resume_version_id=ats_resume_version_id,
            format=format,
            s3_bucket=s3_bucket,
            s3_key=s3_key,
            file_size_bytes=file_size_bytes,
        )
        self.session.add(file_row)
        await self.session.flush()
        return file_row

    # --- cross-module reads ------------------------------------------------------

    async def get_source_resume_version_owned(
        self, *, resume_version_id: uuid.UUID, user_id: uuid.UUID
    ) -> ResumeVersion | None:
        stmt = (
            select(ResumeVersion)
            .join(Resume, Resume.id == ResumeVersion.resume_id)
            .where(
                ResumeVersion.id == resume_version_id,
                Resume.user_id == user_id,
                ResumeVersion.deleted_at.is_(None),
            )
            .options(selectinload(ResumeVersion.parsed_data))
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def get_source_resume_version(self, *, resume_version_id: uuid.UUID) -> ResumeVersion | None:
        """Fetch a source resume version by id alone (no owner check) — used
        by the Celery task path.
        """
        stmt = (
            select(ResumeVersion)
            .where(ResumeVersion.id == resume_version_id, ResumeVersion.deleted_at.is_(None))
            .options(selectinload(ResumeVersion.parsed_data))
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def job_exists(self, *, job_id: uuid.UUID) -> bool:
        stmt = select(Job.id).where(Job.id == job_id, Job.deleted_at.is_(None))
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none() is not None


__all__ = ["ATSResumeRepository", "ATSResumeVersionRepository"]
