"""Resume Parser module — data access layer."""
from __future__ import annotations

import uuid
from typing import Sequence

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.repositories.base import BaseRepository
from app.modules.resume_parser.models import Resume, ResumeVersion


class ResumeRepository(BaseRepository[Resume]):
    """Data access for the logical `Resume` entity."""

    model = Resume

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def list_by_user(
        self, *, user_id: uuid.UUID, offset: int = 0, limit: int = 20
    ) -> Sequence[Resume]:
        stmt = (
            select(Resume)
            .where(Resume.user_id == user_id, Resume.deleted_at.is_(None))
            .options(selectinload(Resume.versions))
            .order_by(Resume.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().all()

    async def count_by_user(self, *, user_id: uuid.UUID) -> int:
        stmt = select(func.count()).select_from(Resume).where(
            Resume.user_id == user_id, Resume.deleted_at.is_(None)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def get_owned(self, *, resume_id: uuid.UUID, user_id: uuid.UUID) -> Resume | None:
        stmt = (
            select(Resume)
            .where(Resume.id == resume_id, Resume.user_id == user_id, Resume.deleted_at.is_(None))
            .options(selectinload(Resume.versions))
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().one_or_none()


class ResumeVersionRepository(BaseRepository[ResumeVersion]):
    """Data access for `ResumeVersion`, including current-version bookkeeping."""

    model = ResumeVersion

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def list_for_resume(self, *, resume_id: uuid.UUID) -> Sequence[ResumeVersion]:
        stmt = (
            select(ResumeVersion)
            .where(ResumeVersion.resume_id == resume_id, ResumeVersion.deleted_at.is_(None))
            .order_by(ResumeVersion.version_number.desc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_current(self, *, resume_id: uuid.UUID) -> ResumeVersion | None:
        stmt = select(ResumeVersion).where(
            ResumeVersion.resume_id == resume_id,
            ResumeVersion.is_current.is_(True),
            ResumeVersion.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def get_for_resume(
        self, *, resume_id: uuid.UUID, version_id: uuid.UUID, with_details: bool = False
    ) -> ResumeVersion | None:
        stmt = select(ResumeVersion).where(
            ResumeVersion.id == version_id,
            ResumeVersion.resume_id == resume_id,
            ResumeVersion.deleted_at.is_(None),
        )
        if with_details:
            stmt = stmt.options(
                selectinload(ResumeVersion.parsed_data), selectinload(ResumeVersion.embedding)
            )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def next_version_number(self, *, resume_id: uuid.UUID) -> int:
        stmt = select(func.max(ResumeVersion.version_number)).where(
            ResumeVersion.resume_id == resume_id
        )
        result = await self.session.execute(stmt)
        current_max = result.scalar_one_or_none()
        return (current_max or 0) + 1

    async def set_current(self, *, resume_id: uuid.UUID, version_id: uuid.UUID) -> None:
        """Atomically make `version_id` the only current version of `resume_id`."""
        await self.session.execute(
            update(ResumeVersion)
            .where(ResumeVersion.resume_id == resume_id, ResumeVersion.id != version_id)
            .values(is_current=False)
        )
        await self.session.execute(
            update(ResumeVersion).where(ResumeVersion.id == version_id).values(is_current=True)
        )
        await self.session.flush()
