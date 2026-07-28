"""Daily AI Agent module — data access layer."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.daily_agent.models import DailyAgentRun
from app.repositories.base import BaseRepository


class DailyAgentRunRepository(BaseRepository[DailyAgentRun]):
    model = DailyAgentRun

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_latest_for_user(self, *, user_id: uuid.UUID) -> DailyAgentRun | None:
        stmt = (
            select(DailyAgentRun)
            .where(DailyAgentRun.user_id == user_id)
            .order_by(DailyAgentRun.started_at.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def list_recent_for_user(
        self, *, user_id: uuid.UUID, limit: int = 10
    ) -> Sequence[DailyAgentRun]:
        stmt = (
            select(DailyAgentRun)
            .where(DailyAgentRun.user_id == user_id)
            .order_by(DailyAgentRun.started_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()


__all__ = ["DailyAgentRunRepository"]
