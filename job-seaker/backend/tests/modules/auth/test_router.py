"""Integration tests for the `/auth` API — register, login, refresh
rotation, logout, and the `/me` identity endpoint.

Needs a reachable Postgres with the auth tables migrated:

    docker compose up -d postgres redis
    docker compose exec api alembic upgrade head
    docker compose exec api pytest tests/modules/auth/test_router.py
"""
from __future__ import annotations

import uuid

from httpx import AsyncClient

from app.core.security import REFRESH_COOKIE_NAME


def _unique_email() -> str:
    return f"{uuid.uuid4().hex}@example.com"


async def test_register_returns_access_token_and_sets_refresh_cookie(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": _unique_email(), "password": "correct horse battery staple"},
    )
    assert response.status_code == 201
    body = response.json()["data"]
    assert body["access_token"]
    assert body["user"]["is_active"] is True
    assert REFRESH_COOKIE_NAME in response.cookies


async def test_register_rejects_duplicate_email(client: AsyncClient) -> None:
    email = _unique_email()
    payload = {"email": email, "password": "correct horse battery staple"}
    first = await client.post("/api/v1/auth/register", json=payload)
    assert first.status_code == 201

    second = await client.post("/api/v1/auth/register", json=payload)
    assert second.status_code == 409


async def test_register_rejects_short_password(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/auth/register", json={"email": _unique_email(), "password": "short"}
    )
    assert response.status_code == 422


async def test_login_with_correct_credentials_succeeds(client: AsyncClient) -> None:
    email = _unique_email()
    await client.post(
        "/api/v1/auth/register", json={"email": email, "password": "correct horse battery staple"}
    )
    response = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": "correct horse battery staple"}
    )
    assert response.status_code == 200
    assert response.json()["data"]["access_token"]


async def test_login_with_wrong_password_returns_401(client: AsyncClient) -> None:
    email = _unique_email()
    await client.post(
        "/api/v1/auth/register", json={"email": email, "password": "correct horse battery staple"}
    )
    response = await client.post("/api/v1/auth/login", json={"email": email, "password": "nope"})
    assert response.status_code == 401


async def test_me_requires_bearer_token(client: AsyncClient) -> None:
    response = await client.get("/api/v1/auth/me")
    assert response.status_code == 401


async def test_me_returns_the_authenticated_user(client: AsyncClient) -> None:
    email = _unique_email()
    register = await client.post(
        "/api/v1/auth/register", json={"email": email, "password": "correct horse battery staple"}
    )
    access_token = register.json()["data"]["access_token"]

    response = await client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {access_token}"}
    )
    assert response.status_code == 200
    assert response.json()["data"]["email"] == email


async def test_refresh_rotates_the_cookie_and_issues_a_new_access_token(
    client: AsyncClient,
) -> None:
    email = _unique_email()
    register = await client.post(
        "/api/v1/auth/register", json={"email": email, "password": "correct horse battery staple"}
    )
    first_access_token = register.json()["data"]["access_token"]
    first_refresh_cookie = client.cookies.get(REFRESH_COOKIE_NAME)

    refresh = await client.post("/api/v1/auth/refresh")
    assert refresh.status_code == 200
    assert refresh.json()["data"]["access_token"] != first_access_token
    assert client.cookies.get(REFRESH_COOKIE_NAME) != first_refresh_cookie


async def test_refresh_without_a_cookie_returns_401(client: AsyncClient) -> None:
    response = await client.post("/api/v1/auth/refresh")
    assert response.status_code == 401


async def test_logout_revokes_the_session_so_refresh_then_fails(client: AsyncClient) -> None:
    email = _unique_email()
    await client.post(
        "/api/v1/auth/register", json={"email": email, "password": "correct horse battery staple"}
    )

    logout = await client.post("/api/v1/auth/logout")
    assert logout.status_code == 200

    refresh = await client.post("/api/v1/auth/refresh")
    assert refresh.status_code == 401
