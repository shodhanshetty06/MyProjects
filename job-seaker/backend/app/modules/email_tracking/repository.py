"""Email Tracking module — data access layer."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.email_tracking.models import (
    GmailAccount,
    GmailAccountStatus,
    TrackedEmail,
)
from app.modules.email_tracking.schemas import TrackedEmailFilters
from app.repositories.base import BaseRepository


class GmailAccountRepository(BaseRepository[GmailAccount]):
    model = GmailAccount

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_by_email(self, *, email_address: str) -> GmailAccount | None:
        stmt = select(GmailAccount).where(GmailAccount.email_address == email_address)
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def get_owned(self, *, account_id: uuid.UUID, user_id: uuid.UUID) -> GmailAccount | None:
        stmt = select(GmailAccount).where(
            GmailAccount.id == account_id, GmailAccount.user_id == user_id
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def list_by_user(self, *, user_id: uuid.UUID) -> Sequence[GmailAccount]:
        stmt = select(GmailAccount).where(GmailAccount.user_id == user_id)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def list_connected(self) -> Sequence[GmailAccount]:
        stmt = select(GmailAccount).where(GmailAccount.status == GmailAccountStatus.CONNECTED)
        result = await self.session.execute(stmt)
        return result.scalars().all()


class TrackedEmailRepository(BaseRepository[TrackedEmail]):
    model = TrackedEmail

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_by_message_id(
        self, *, gmail_account_id: uuid.UUID, gmail_message_id: str
    ) -> TrackedEmail | None:
        stmt = select(TrackedEmail).where(
            TrackedEmail.gmail_account_id == gmail_account_id,
            TrackedEmail.gmail_message_id == gmail_message_id,
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def get_owned(self, *, email_id: uuid.UUID, user_id: uuid.UUID) -> TrackedEmail | None:
        stmt = (
            select(TrackedEmail)
            .join(GmailAccount, TrackedEmail.gmail_account_id == GmailAccount.id)
            .where(TrackedEmail.id == email_id, GmailAccount.user_id == user_id)
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    def _apply_filters(self, stmt, *, user_id: uuid.UUID, filters: TrackedEmailFilters):
        stmt = stmt.join(GmailAccount, TrackedEmail.gmail_account_id == GmailAccount.id).where(
            GmailAccount.user_id == user_id
        )
        if filters.match_status:
            stmt = stmt.where(TrackedEmail.match_status == filters.match_status)
        if filters.application_id:
            stmt = stmt.where(TrackedEmail.application_id == filters.application_id)
        return stmt

    async def list_filtered(
        self, *, user_id: uuid.UUID, filters: TrackedEmailFilters, offset: int, limit: int
    ) -> Sequence[TrackedEmail]:
        stmt = self._apply_filters(select(TrackedEmail), user_id=user_id, filters=filters)
        stmt = stmt.order_by(TrackedEmail.received_at.desc()).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def count_filtered(self, *, user_id: uuid.UUID, filters: TrackedEmailFilters) -> int:
        stmt = self._apply_filters(
            select(func.count(TrackedEmail.id)), user_id=user_id, filters=filters
        )
        result = await self.session.execute(stmt)
        return result.scalar_one()


__all__ = ["GmailAccountRepository", "TrackedEmailRepository"]
