"""User Profile module — data access layer."""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.user_profile.models import UserProfile
from app.repositories.base import BaseRepository


class UserProfileRepository(BaseRepository[UserProfile]):
    """Data access for the onboarding-only `UserProfile` row."""

    model = UserProfile

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_by_user(self, *, user_id: uuid.UUID) -> UserProfile | None:
        stmt = select(UserProfile).where(UserProfile.user_id == user_id)
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def upsert(self, *, user_id: uuid.UUID, **fields: Any) -> UserProfile:
        profile = await self.get_by_user(user_id=user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id, **fields)
            self.session.add(profile)
        else:
            for key, value in fields.items():
                setattr(profile, key, value)
        await self.session.flush()
        return profile
