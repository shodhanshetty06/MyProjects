"""User Profile module — data access layer."""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.base import BaseRepository


class UserProfileRepository(BaseRepository):
    """Data access for the User Profile module.

    Phase 0: no model is bound yet. Once User Profile models exist, set
    `model = <ORMModel>` and add module-specific queries here.
    """

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)
