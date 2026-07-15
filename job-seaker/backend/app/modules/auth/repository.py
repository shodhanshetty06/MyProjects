"""Authentication module — data access layer."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.auth.models import RefreshToken, User
from app.repositories.base import BaseRepository


class AuthRepository(BaseRepository[User]):
    model = User

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, model=User)

    async def get_by_email(self, email: str) -> User | None:
        stmt = select(User).where(User.email == email.lower())
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create_refresh_token(
        self, *, user_id: uuid.UUID, token_hash: str, expires_at: datetime
    ) -> RefreshToken:
        token = RefreshToken(user_id=user_id, token_hash=token_hash, expires_at=expires_at)
        self.session.add(token)
        await self.session.flush()
        return token

    async def get_refresh_token_by_hash(self, token_hash: str) -> RefreshToken | None:
        stmt = select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def revoke_refresh_token(self, token: RefreshToken, *, when: datetime) -> None:
        token.revoked_at = when
        await self.session.flush()

    async def revoke_all_for_user(self, user_id: uuid.UUID, *, when: datetime) -> None:
        stmt = (
            update(RefreshToken)
            .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=when)
        )
        await self.session.execute(stmt)


__all__ = ["AuthRepository"]
