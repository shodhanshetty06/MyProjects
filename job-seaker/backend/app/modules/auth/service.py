"""Authentication module — application/use-case layer."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from app.core.config import get_settings
from app.core.exceptions import ConflictError, ForbiddenError, UnauthorizedError
from app.core.logging import get_logger
from app.core.security import (
    create_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)
from app.modules.auth.models import User
from app.modules.auth.repository import AuthRepository

logger = get_logger(__name__)


class IssuedTokens:
    """Bundles a fresh access/refresh token pair for one authenticated user."""

    __slots__ = ("user", "access_token", "access_token_expires_at", "refresh_token")

    def __init__(
        self,
        *,
        user: User,
        access_token: str,
        access_token_expires_at: datetime,
        refresh_token: str,
    ) -> None:
        self.user = user
        self.access_token = access_token
        self.access_token_expires_at = access_token_expires_at
        self.refresh_token = refresh_token


class AuthService:
    """Business logic orchestration for the Authentication module."""

    def __init__(self, repository: AuthRepository) -> None:
        self.repository = repository
        self.settings = get_settings()

    async def register(self, *, email: str, password: str) -> User:
        normalized_email = email.lower()
        existing = await self.repository.get_by_email(normalized_email)
        if existing is not None:
            raise ConflictError("An account with this email already exists.")
        user = await self.repository.create(
            email=normalized_email, hashed_password=hash_password(password)
        )
        logger.info("User registered", extra={"extra_fields": {"user_id": str(user.id)}})
        return user

    async def authenticate(self, *, email: str, password: str) -> User:
        user = await self.repository.get_by_email(email.lower())
        if user is None or not verify_password(password, user.hashed_password):
            raise UnauthorizedError("Invalid email or password.")
        if not user.is_active:
            raise ForbiddenError("This account has been deactivated.")
        return user

    async def issue_tokens(self, user: User) -> IssuedTokens:
        access_token, expires_at = create_access_token(user.id)
        raw_refresh, refresh_hash = generate_refresh_token()
        refresh_expires_at = datetime.now(timezone.utc) + timedelta(
            days=self.settings.REFRESH_TOKEN_EXPIRE_DAYS
        )
        await self.repository.create_refresh_token(
            user_id=user.id, token_hash=refresh_hash, expires_at=refresh_expires_at
        )
        return IssuedTokens(
            user=user,
            access_token=access_token,
            access_token_expires_at=expires_at,
            refresh_token=raw_refresh,
        )

    async def refresh(self, raw_refresh_token: str) -> IssuedTokens:
        token_hash = hash_refresh_token(raw_refresh_token)
        stored = await self.repository.get_refresh_token_by_hash(token_hash)
        now = datetime.now(timezone.utc)
        if stored is None or stored.revoked_at is not None or stored.expires_at < now:
            raise UnauthorizedError("Refresh token is invalid or expired.")

        user = await self.repository.get(stored.user_id)
        if user is None or not user.is_active:
            raise UnauthorizedError("Refresh token is invalid or expired.")

        # Rotate on every use: the old token is dead the instant a new one is
        # minted, so a stolen-then-replayed refresh token is only ever
        # usable once before the legitimate client's next refresh fails loudly.
        await self.repository.revoke_refresh_token(stored, when=now)
        return await self.issue_tokens(user)

    async def logout(self, raw_refresh_token: str | None) -> None:
        if not raw_refresh_token:
            return
        token_hash = hash_refresh_token(raw_refresh_token)
        stored = await self.repository.get_refresh_token_by_hash(token_hash)
        if stored is not None and stored.revoked_at is None:
            await self.repository.revoke_refresh_token(stored, when=datetime.now(timezone.utc))

    async def logout_all(self, user_id: uuid.UUID) -> None:
        await self.repository.revoke_all_for_user(user_id, when=datetime.now(timezone.utc))

    async def get_user(self, user_id: uuid.UUID) -> User:
        user = await self.repository.get(user_id)
        if user is None:
            raise UnauthorizedError("Account no longer exists.")
        return user


__all__ = ["AuthService", "IssuedTokens"]
