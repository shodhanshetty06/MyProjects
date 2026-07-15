"""Tracking module — data access layer."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import Select, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.modules.tracking.models import (
    Application,
    ApplicationNote,
    ApplicationStatus,
    ApplicationStatusHistory,
)
from app.modules.tracking.schemas import ApplicationFilters, ApplicationSortBy, SortDirection
from app.repositories.base import BaseRepository

_DETAIL_EAGER_LOAD = (
    selectinload(Application.notes),
    selectinload(Application.status_history),
)


def _sql_order_by(sort_by: ApplicationSortBy, sort_direction: SortDirection) -> tuple:
    descending = sort_direction == SortDirection.DESC
    column = {
        ApplicationSortBy.APPLIED_AT: Application.applied_at,
        ApplicationSortBy.UPDATED_AT: Application.updated_at,
        ApplicationSortBy.COMPANY_NAME: Application.company_name,
        ApplicationSortBy.JOB_TITLE: Application.job_title,
        ApplicationSortBy.STATUS: Application.status,
    }[sort_by]
    return (column.desc() if descending else column.asc(),)


class ApplicationRepository(BaseRepository[Application]):
    model = Application

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    def _apply_filters(
        self, stmt: Select, *, user_id: uuid.UUID, filters: ApplicationFilters
    ) -> Select:
        stmt = stmt.where(Application.user_id == user_id, Application.deleted_at.is_(None))
        if filters.status:
            stmt = stmt.where(Application.status.in_(filters.status))
        if filters.company_name:
            stmt = stmt.where(Application.company_name.ilike(f"%{filters.company_name}%"))
        if filters.job_id:
            stmt = stmt.where(Application.job_id == filters.job_id)
        if filters.q:
            like = f"%{filters.q}%"
            stmt = stmt.where(
                or_(Application.company_name.ilike(like), Application.job_title.ilike(like))
            )
        if filters.applied_after is not None:
            stmt = stmt.where(Application.applied_at >= filters.applied_after)
        if filters.applied_before is not None:
            stmt = stmt.where(Application.applied_at <= filters.applied_before)
        return stmt

    async def get_owned(
        self, *, application_id: uuid.UUID, user_id: uuid.UUID, with_details: bool = False
    ) -> Application | None:
        stmt = select(Application).where(
            Application.id == application_id,
            Application.user_id == user_id,
            Application.deleted_at.is_(None),
        )
        if with_details:
            stmt = stmt.options(*_DETAIL_EAGER_LOAD)
        result = await self.session.execute(stmt)
        return result.scalars().unique().one_or_none()

    async def list_filtered(
        self, *, user_id: uuid.UUID, filters: ApplicationFilters, offset: int, limit: int
    ) -> Sequence[Application]:
        stmt = self._apply_filters(select(Application), user_id=user_id, filters=filters)
        stmt = (
            stmt.order_by(*_sql_order_by(filters.sort_by, filters.sort_direction))
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().all()

    async def count_filtered(self, *, user_id: uuid.UUID, filters: ApplicationFilters) -> int:
        stmt = self._apply_filters(
            select(func.count(Application.id)), user_id=user_id, filters=filters
        )
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def status_counts(self, *, user_id: uuid.UUID) -> dict[str, int]:
        stmt = (
            select(Application.status, func.count(Application.id))
            .where(Application.user_id == user_id, Application.deleted_at.is_(None))
            .group_by(Application.status)
        )
        rows = (await self.session.execute(stmt)).all()
        return {row[0]: row[1] for row in rows}

    async def count_since(self, *, user_id: uuid.UUID, since: datetime) -> int:
        stmt = select(func.count(Application.id)).where(
            Application.user_id == user_id,
            Application.deleted_at.is_(None),
            Application.applied_at >= since,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def list_recent(self, *, user_id: uuid.UUID, limit: int) -> Sequence[Application]:
        stmt = (
            select(Application)
            .where(Application.user_id == user_id, Application.deleted_at.is_(None))
            .order_by(Application.updated_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def list_needing_follow_up(
        self, *, user_id: uuid.UUID, stale_before: datetime, limit: int
    ) -> Sequence[Application]:
        """Applications still `applied` with no activity since `stale_before`."""
        stmt = (
            select(Application)
            .where(
                Application.user_id == user_id,
                Application.deleted_at.is_(None),
                Application.status == ApplicationStatus.APPLIED,
                Application.updated_at < stale_before,
            )
            .order_by(Application.updated_at.asc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def average_days_to_first_response(self, *, user_id: uuid.UUID) -> float | None:
        """Average time between `applied_at` and an application's first status change."""
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
        result = await self.session.execute(stmt)
        seconds = result.scalar_one_or_none()
        if seconds is None:
            return None
        return round(float(seconds) / 86400, 2)


class ApplicationStatusHistoryRepository(BaseRepository[ApplicationStatusHistory]):
    model = ApplicationStatusHistory

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def list_by_application(
        self, *, application_id: uuid.UUID
    ) -> Sequence[ApplicationStatusHistory]:
        stmt = (
            select(ApplicationStatusHistory)
            .where(ApplicationStatusHistory.application_id == application_id)
            .order_by(ApplicationStatusHistory.created_at.asc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()


class ApplicationNoteRepository(BaseRepository[ApplicationNote]):
    model = ApplicationNote

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_for_application(
        self, *, note_id: uuid.UUID, application_id: uuid.UUID
    ) -> ApplicationNote | None:
        stmt = select(ApplicationNote).where(
            ApplicationNote.id == note_id, ApplicationNote.application_id == application_id
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def list_by_application(
        self, *, application_id: uuid.UUID
    ) -> Sequence[ApplicationNote]:
        stmt = (
            select(ApplicationNote)
            .where(ApplicationNote.application_id == application_id)
            .order_by(ApplicationNote.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()
