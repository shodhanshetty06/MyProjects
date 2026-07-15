"""Gmail OAuth2 client — authorization URL, code exchange, token refresh, revoke.

Talks to Google's OAuth endpoints via plain `httpx` rather than the
`google-auth`/`google-api-python-client` SDKs, matching the codebase's
existing convention of hitting third-party HTTP APIs directly (see every
`job_ingestion` ATS adapter and the `auto_apply` HTTP-platform adapters).

A fresh `httpx.AsyncClient` is built per call rather than cached/injected,
mirroring `app/integrations/http_client.py`'s reasoning: Celery tasks run via
`asyncio.run(...)` per invocation (a new event loop each time), and an
`httpx.AsyncClient`'s connection pool is bound to the loop it was created on.
"""

from __future__ import annotations

from urllib.parse import urlencode

import httpx
from pydantic import BaseModel

from app.core.config import get_settings
from app.modules.email_tracking.errors import GmailAuthError

AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"


class GmailTokenResponse(BaseModel):
    access_token: str
    refresh_token: str | None = None
    expires_in: int
    scope: str


class GmailOAuthClient:
    def __init__(self) -> None:
        self._settings = get_settings()

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self._settings.GMAIL_API_TIMEOUT_SECONDS)

    def build_authorization_url(self, *, state: str) -> str:
        params = {
            "client_id": self._settings.GMAIL_OAUTH_CLIENT_ID,
            "redirect_uri": self._settings.GMAIL_OAUTH_REDIRECT_URI,
            "response_type": "code",
            "scope": self._settings.GMAIL_OAUTH_SCOPES,
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }
        return f"{AUTHORIZATION_ENDPOINT}?{urlencode(params)}"

    async def exchange_code_for_tokens(self, *, code: str) -> GmailTokenResponse:
        async with self._client() as client:
            response = await client.post(
                TOKEN_ENDPOINT,
                data={
                    "code": code,
                    "client_id": self._settings.GMAIL_OAUTH_CLIENT_ID,
                    "client_secret": self._settings.GMAIL_OAUTH_CLIENT_SECRET,
                    "redirect_uri": self._settings.GMAIL_OAUTH_REDIRECT_URI,
                    "grant_type": "authorization_code",
                },
            )
        _raise_for_auth_error(response)
        response.raise_for_status()
        return GmailTokenResponse.model_validate(response.json())

    async def refresh_access_token(self, *, refresh_token: str) -> GmailTokenResponse:
        async with self._client() as client:
            response = await client.post(
                TOKEN_ENDPOINT,
                data={
                    "refresh_token": refresh_token,
                    "client_id": self._settings.GMAIL_OAUTH_CLIENT_ID,
                    "client_secret": self._settings.GMAIL_OAUTH_CLIENT_SECRET,
                    "grant_type": "refresh_token",
                },
            )
        _raise_for_auth_error(response)
        response.raise_for_status()
        payload = response.json()
        payload.setdefault("refresh_token", refresh_token)
        return GmailTokenResponse.model_validate(payload)

    async def revoke_token(self, *, token: str) -> None:
        """Best-effort revoke — failures are not raised, since disconnect
        should always succeed locally even if Google's revoke call fails."""
        try:
            async with self._client() as client:
                await client.post(REVOKE_ENDPOINT, data={"token": token})
        except httpx.HTTPError:
            pass


def _raise_for_auth_error(response: httpx.Response) -> None:
    """Google returns HTTP 400 (not 401) with `error=invalid_grant` when a
    refresh token has been revoked or expired — map that explicitly to
    `GmailAuthError` so it isn't misclassified as a generic/transient 400."""
    if response.status_code != 400:
        return
    try:
        error = response.json().get("error")
    except ValueError:
        return
    if error in ("invalid_grant", "unauthorized_client"):
        raise GmailAuthError(f"Gmail OAuth error: {error}")


__all__ = ["GmailOAuthClient", "GmailTokenResponse"]
