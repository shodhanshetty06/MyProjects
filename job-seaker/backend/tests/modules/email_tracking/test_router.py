"""Integration tests for the `/email-tracking` API — Gmail connect/callback,
account listing/disconnect, and the tracked-email review queue.

`GmailOAuthClient`/`GmailAPIClient` are monkeypatched at the class level (the
router constructs a fresh instance per request via dependency injection, so
patching the class method is what actually takes effect for every instance),
matching the "monkeypatch provider client" convention used elsewhere in the
suite. No real Google OAuth/network access is required.

Needs a reachable Postgres + Redis with migrations applied:

    docker compose up -d postgres redis
    docker compose exec api alembic upgrade head
    docker compose exec api pytest tests/modules/email_tracking/test_router.py
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from httpx import AsyncClient

from app.core.security import create_access_token
from app.db.session import AsyncSessionLocal
from app.modules.email_tracking.gmail_api_client import GmailAPIClient, GmailProfile
from app.modules.email_tracking.gmail_oauth_client import GmailOAuthClient, GmailTokenResponse
from app.modules.email_tracking.models import (
    EmailClassification,
    EmailMatchStatus,
    GmailAccount,
    GmailAccountStatus,
    TrackedEmail,
)
from app.modules.tracking.models import Application, ApplicationSource, ApplicationStatus
from app.modules.tracking.repository import (
    ApplicationRepository,
    ApplicationStatusHistoryRepository,
)


def _user_header(user_id: uuid.UUID | None = None) -> dict[str, str]:
    token, _ = create_access_token(user_id or uuid.uuid4())
    return {"Authorization": f"Bearer {token}"}


async def _seed_gmail_account(*, user_id: uuid.UUID) -> GmailAccount:
    async with AsyncSessionLocal() as session:
        account = GmailAccount(
            user_id=user_id,
            email_address=f"{uuid.uuid4().hex}@example.com",
            status=GmailAccountStatus.CONNECTED,
            access_token_encrypted="placeholder",
            refresh_token_encrypted="placeholder",
            token_expires_at=datetime.now(UTC),
            scopes="https://www.googleapis.com/auth/gmail.readonly",
            history_id="1",
        )
        session.add(account)
        await session.commit()
        await session.refresh(account)
        return account


async def _seed_application(*, user_id: uuid.UUID) -> Application:
    async with AsyncSessionLocal() as session:
        application_repo = ApplicationRepository(session)
        history_repo = ApplicationStatusHistoryRepository(session)
        application = await application_repo.create(
            user_id=user_id,
            job_id=None,
            company_name="Acme Corp",
            job_title="Backend Engineer",
            job_url=None,
            location=None,
            source=ApplicationSource.MANUAL,
            status=ApplicationStatus.APPLIED,
            applied_at=datetime.now(UTC),
        )
        await history_repo.create(
            application_id=application.id,
            from_status=None,
            to_status=ApplicationStatus.APPLIED,
            note=None,
        )
        await session.commit()
        await session.refresh(application)
        return application


async def _seed_tracked_email(
    *, account_id: uuid.UUID, match_status: EmailMatchStatus = EmailMatchStatus.NEEDS_REVIEW
) -> TrackedEmail:
    async with AsyncSessionLocal() as session:
        email = TrackedEmail(
            gmail_account_id=account_id,
            application_id=None,
            gmail_message_id=f"msg-{uuid.uuid4().hex}",
            thread_id=None,
            sender="recruiting@acme.com",
            subject="Interview scheduling",
            snippet="Let's schedule your interview.",
            received_at=datetime.now(UTC),
            classification=EmailClassification.INTERVIEW,
            confidence=0.5,
            match_status=match_status,
            matched_status_history_id=None,
            raw_metadata={},
        )
        session.add(email)
        await session.commit()
        await session.refresh(email)
        return email


async def test_connect_returns_authorization_url(client: AsyncClient) -> None:
    response = await client.get("/api/v1/email-tracking/gmail/connect", headers=_user_header())

    assert response.status_code == 200
    url = response.json()["data"]["authorization_url"]
    assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")


async def test_callback_creates_gmail_account(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = uuid.uuid4()

    connect_response = await client.get(
        "/api/v1/email-tracking/gmail/connect", headers=_user_header(user_id)
    )
    authorization_url = connect_response.json()["data"]["authorization_url"]
    state = authorization_url.split("state=")[1].split("&")[0]

    async def _fake_exchange(self, *, code: str) -> GmailTokenResponse:
        return GmailTokenResponse(
            access_token="access-token",
            refresh_token="refresh-token",
            expires_in=3600,
            scope="https://www.googleapis.com/auth/gmail.readonly",
        )

    async def _fake_get_profile(self) -> GmailProfile:
        return GmailProfile(email_address=f"{uuid.uuid4().hex}@example.com", history_id="42")

    monkeypatch.setattr(GmailOAuthClient, "exchange_code_for_tokens", _fake_exchange)
    monkeypatch.setattr(GmailAPIClient, "get_profile", _fake_get_profile)

    callback_response = await client.get(
        "/api/v1/email-tracking/gmail/callback", params={"code": "auth-code", "state": state}
    )

    assert callback_response.status_code == 200
    body = callback_response.json()["data"]
    assert body["status"] == "connected"
    assert body["history_id"] == "42"


async def test_callback_rejects_invalid_state(client: AsyncClient) -> None:
    response = await client.get(
        "/api/v1/email-tracking/gmail/callback",
        params={"code": "auth-code", "state": "not-a-real-state"},
    )
    assert response.status_code == 422


async def test_list_accounts_only_returns_caller_s_accounts(client: AsyncClient) -> None:
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    await _seed_gmail_account(user_id=user_a)
    await _seed_gmail_account(user_id=user_b)

    response = await client.get("/api/v1/email-tracking/accounts", headers=_user_header(user_a))

    assert response.status_code == 200
    assert len(response.json()["data"]) == 1


async def test_get_tracked_email_not_found_for_other_users(client: AsyncClient) -> None:
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    account = await _seed_gmail_account(user_id=user_a)
    email = await _seed_tracked_email(account_id=account.id)

    response = await client.get(
        f"/api/v1/email-tracking/emails/{email.id}", headers=_user_header(user_b)
    )
    assert response.status_code == 404


async def test_resolve_dismiss_marks_ignored(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    account = await _seed_gmail_account(user_id=user_id)
    email = await _seed_tracked_email(account_id=account.id)

    response = await client.post(
        f"/api/v1/email-tracking/emails/{email.id}/resolve",
        headers=_user_header(user_id),
        json={"dismiss": True},
    )

    assert response.status_code == 200
    assert response.json()["data"]["match_status"] == "ignored"


async def test_resolve_confirm_updates_application_status(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    account = await _seed_gmail_account(user_id=user_id)
    application = await _seed_application(user_id=user_id)
    email = await _seed_tracked_email(account_id=account.id)

    response = await client.post(
        f"/api/v1/email-tracking/emails/{email.id}/resolve",
        headers=_user_header(user_id),
        json={"application_id": str(application.id), "status": "interview"},
    )

    assert response.status_code == 200
    body = response.json()["data"]
    assert body["match_status"] == "matched"
    assert body["application_id"] == str(application.id)

    application_response = await client.get(
        f"/api/v1/tracking/applications/{application.id}", headers=_user_header(user_id)
    )
    assert application_response.json()["data"]["status"] == "interview"


async def test_disconnect_account(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    user_id = uuid.uuid4()
    account = await _seed_gmail_account(user_id=user_id)

    async def _fake_revoke(self, *, token: str) -> None:
        return None

    monkeypatch.setattr(GmailOAuthClient, "revoke_token", _fake_revoke)

    response = await client.delete(
        f"/api/v1/email-tracking/accounts/{account.id}", headers=_user_header(user_id)
    )

    assert response.status_code == 200

    async with AsyncSessionLocal() as session:
        refreshed = await session.get(GmailAccount, account.id)
        assert refreshed.status == GmailAccountStatus.DISCONNECTED
