"""Integration tests for `AuthService` against a real DB session.

Needs a reachable Postgres with the auth tables migrated:

    docker compose up -d postgres redis
    docker compose exec api alembic upgrade head
    docker compose exec api pytest tests/modules/auth/test_service.py
"""
from __future__ import annotations

import uuid

import pytest

from app.core.exceptions import ConflictError, ForbiddenError, UnauthorizedError
from app.db.session import AsyncSessionLocal
from app.modules.auth.repository import AuthRepository
from app.modules.auth.service import AuthService


def _build_service(session) -> AuthService:
    return AuthService(AuthRepository(session))


def _unique_email() -> str:
    return f"{uuid.uuid4().hex}@example.com"


async def test_register_creates_user_with_hashed_password() -> None:
    async with AsyncSessionLocal() as session:
        service = _build_service(session)
        email = _unique_email()
        user = await service.register(email=email, password="correct horse battery staple")
        assert user.email == email
        assert user.hashed_password != "correct horse battery staple"
        assert user.is_active is True


async def test_register_rejects_duplicate_email() -> None:
    async with AsyncSessionLocal() as session:
        service = _build_service(session)
        email = _unique_email()
        await service.register(email=email, password="correct horse battery staple")
        with pytest.raises(ConflictError):
            await service.register(email=email, password="another password entirely")


async def test_register_normalizes_email_case_for_duplicate_detection() -> None:
    async with AsyncSessionLocal() as session:
        service = _build_service(session)
        email = _unique_email()
        await service.register(email=email, password="correct horse battery staple")
        with pytest.raises(ConflictError):
            await service.register(email=email.upper(), password="another password entirely")


async def test_authenticate_succeeds_with_correct_credentials() -> None:
    async with AsyncSessionLocal() as session:
        service = _build_service(session)
        email = _unique_email()
        await service.register(email=email, password="correct horse battery staple")
        user = await service.authenticate(email=email, password="correct horse battery staple")
        assert user.email == email


async def test_authenticate_rejects_wrong_password() -> None:
    async with AsyncSessionLocal() as session:
        service = _build_service(session)
        email = _unique_email()
        await service.register(email=email, password="correct horse battery staple")
        with pytest.raises(UnauthorizedError):
            await service.authenticate(email=email, password="wrong password")


async def test_authenticate_rejects_unknown_email() -> None:
    async with AsyncSessionLocal() as session:
        service = _build_service(session)
        with pytest.raises(UnauthorizedError):
            await service.authenticate(email=_unique_email(), password="whatever")


async def test_authenticate_rejects_inactive_user() -> None:
    async with AsyncSessionLocal() as session:
        service = _build_service(session)
        email = _unique_email()
        user = await service.register(email=email, password="correct horse battery staple")
        await service.repository.update(user, is_active=False)
        with pytest.raises(ForbiddenError):
            await service.authenticate(email=email, password="correct horse battery staple")


async def test_refresh_rotates_token_and_old_one_becomes_unusable() -> None:
    async with AsyncSessionLocal() as session:
        service = _build_service(session)
        user = await service.register(email=_unique_email(), password="correct horse battery staple")
        first = await service.issue_tokens(user)

        second = await service.refresh(first.refresh_token)
        assert second.access_token != first.access_token
        assert second.refresh_token != first.refresh_token

        with pytest.raises(UnauthorizedError):
            await service.refresh(first.refresh_token)


async def test_refresh_rejects_unknown_token() -> None:
    async with AsyncSessionLocal() as session:
        service = _build_service(session)
        with pytest.raises(UnauthorizedError):
            await service.refresh("not-a-real-refresh-token")


async def test_logout_revokes_refresh_token() -> None:
    async with AsyncSessionLocal() as session:
        service = _build_service(session)
        user = await service.register(email=_unique_email(), password="correct horse battery staple")
        tokens = await service.issue_tokens(user)

        await service.logout(tokens.refresh_token)

        with pytest.raises(UnauthorizedError):
            await service.refresh(tokens.refresh_token)


async def test_logout_all_revokes_every_refresh_token_for_the_user() -> None:
    async with AsyncSessionLocal() as session:
        service = _build_service(session)
        user = await service.register(email=_unique_email(), password="correct horse battery staple")
        first = await service.issue_tokens(user)
        second = await service.issue_tokens(user)

        await service.logout_all(user.id)

        with pytest.raises(UnauthorizedError):
            await service.refresh(first.refresh_token)
        with pytest.raises(UnauthorizedError):
            await service.refresh(second.refresh_token)
