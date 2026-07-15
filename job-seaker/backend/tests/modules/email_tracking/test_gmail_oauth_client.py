"""Unit tests for `GmailOAuthClient` — each call fed a mocked `httpx`
transport (`httpx.MockTransport`) so no real network access is required,
mirroring `tests/modules/job_ingestion/test_adapters.py`'s convention.
"""

from __future__ import annotations

import httpx
import pytest

from app.core.config import get_settings
from app.modules.email_tracking.errors import GmailAuthError
from app.modules.email_tracking.gmail_oauth_client import GmailOAuthClient


@pytest.fixture(autouse=True)
def _oauth_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("GMAIL_OAUTH_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("GMAIL_OAUTH_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv(
        "GMAIL_OAUTH_REDIRECT_URI", "http://localhost:8000/api/v1/email-tracking/gmail/callback"
    )
    yield
    get_settings.cache_clear()


def _patch_client(monkeypatch: pytest.MonkeyPatch, client: GmailOAuthClient, handler) -> None:
    monkeypatch.setattr(
        client, "_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


def test_build_authorization_url_includes_client_id_and_state() -> None:
    client = GmailOAuthClient()
    url = client.build_authorization_url(state="abc123")

    assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    assert "client_id=test-client-id" in url
    assert "state=abc123" in url
    assert "access_type=offline" in url


@pytest.mark.asyncio
async def test_exchange_code_for_tokens_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/token"
        return httpx.Response(
            200,
            json={
                "access_token": "access-123",
                "refresh_token": "refresh-456",
                "expires_in": 3599,
                "scope": "https://www.googleapis.com/auth/gmail.readonly",
            },
        )

    client = GmailOAuthClient()
    _patch_client(monkeypatch, client, handler)

    tokens = await client.exchange_code_for_tokens(code="auth-code")

    assert tokens.access_token == "access-123"
    assert tokens.refresh_token == "refresh-456"
    assert tokens.expires_in == 3599


@pytest.mark.asyncio
async def test_refresh_access_token_preserves_original_refresh_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"access_token": "new-access", "expires_in": 3599, "scope": "gmail.readonly"},
        )

    client = GmailOAuthClient()
    _patch_client(monkeypatch, client, handler)

    tokens = await client.refresh_access_token(refresh_token="original-refresh")

    assert tokens.access_token == "new-access"
    assert tokens.refresh_token == "original-refresh"


@pytest.mark.asyncio
async def test_refresh_access_token_raises_gmail_auth_error_on_invalid_grant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant"})

    client = GmailOAuthClient()
    _patch_client(monkeypatch, client, handler)

    with pytest.raises(GmailAuthError):
        await client.refresh_access_token(refresh_token="revoked-token")


@pytest.mark.asyncio
async def test_revoke_token_swallows_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = GmailOAuthClient()
    _patch_client(monkeypatch, client, handler)

    await client.revoke_token(token="whatever")  # must not raise
