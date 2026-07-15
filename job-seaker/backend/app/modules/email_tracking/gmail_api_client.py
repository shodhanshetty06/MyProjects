"""Gmail REST API client — profile, incremental history sync, message fetch.

Plain `httpx` against the Gmail v1 REST API — see `gmail_oauth_client.py`'s
module docstring for why no Google SDK is used, and why a fresh
`httpx.AsyncClient` is built per call rather than cached.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from app.core.config import get_settings
from app.modules.email_tracking.errors import GmailHistoryExpiredError

BASE_URL = "https://gmail.googleapis.com/gmail/v1/users/me"


@dataclass
class GmailMessage:
    message_id: str
    thread_id: str
    sender: str
    subject: str
    snippet: str
    received_at: datetime
    label_ids: list[str]


@dataclass
class GmailHistoryPage:
    message_ids: list[str]
    next_history_id: str


@dataclass
class GmailProfile:
    email_address: str
    history_id: str


class GmailAPIClient:
    def __init__(self, *, access_token: str) -> None:
        self._access_token = access_token
        self._settings = get_settings()

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self._settings.GMAIL_API_TIMEOUT_SECONDS,
            headers={"Authorization": f"Bearer {self._access_token}"},
        )

    async def get_profile(self) -> GmailProfile:
        """The mailbox's address and current `historyId` (used to seed/reseed the sync cursor)."""
        async with self._client() as client:
            response = await client.get(f"{BASE_URL}/profile")
        response.raise_for_status()
        payload = response.json()
        return GmailProfile(
            email_address=payload["emailAddress"], history_id=str(payload["historyId"])
        )

    async def list_history(self, *, start_history_id: str, max_results: int) -> GmailHistoryPage:
        message_ids: list[str] = []
        next_history_id = start_history_id
        page_token: str | None = None

        async with self._client() as client:
            while True:
                params: dict[str, str | int] = {
                    "startHistoryId": start_history_id,
                    "historyTypes": "messageAdded",
                    "labelId": "INBOX",
                    "maxResults": max_results,
                }
                if page_token:
                    params["pageToken"] = page_token
                response = await client.get(f"{BASE_URL}/history", params=params)
                if response.status_code == 404:
                    raise GmailHistoryExpiredError("Gmail history_id is too stale to resume from.")
                response.raise_for_status()
                payload = response.json()

                for entry in payload.get("history", []):
                    for added in entry.get("messagesAdded", []):
                        message_id = added.get("message", {}).get("id")
                        if message_id and message_id not in message_ids:
                            message_ids.append(message_id)
                        if len(message_ids) >= max_results:
                            break
                    if len(message_ids) >= max_results:
                        break

                if "historyId" in payload:
                    next_history_id = str(payload["historyId"])

                page_token = payload.get("nextPageToken")
                if not page_token or len(message_ids) >= max_results:
                    break

        return GmailHistoryPage(message_ids=message_ids, next_history_id=next_history_id)

    async def list_message_ids(self, *, query: str, max_results: int) -> list[str]:
        async with self._client() as client:
            response = await client.get(
                f"{BASE_URL}/messages", params={"q": query, "maxResults": max_results}
            )
        response.raise_for_status()
        return [message["id"] for message in response.json().get("messages", [])]

    async def get_message(self, *, message_id: str) -> GmailMessage:
        async with self._client() as client:
            response = await client.get(
                f"{BASE_URL}/messages/{message_id}",
                params={
                    "format": "metadata",
                    "metadataHeaders": ["Subject", "From"],
                },
            )
        response.raise_for_status()
        payload = response.json()

        headers = {
            header["name"].lower(): header["value"]
            for header in payload.get("payload", {}).get("headers", [])
        }
        internal_date_ms = int(payload.get("internalDate", "0"))
        received_at = datetime.fromtimestamp(internal_date_ms / 1000, tz=UTC)

        return GmailMessage(
            message_id=payload["id"],
            thread_id=payload.get("threadId", payload["id"]),
            sender=headers.get("from", ""),
            subject=headers.get("subject", ""),
            snippet=payload.get("snippet", ""),
            received_at=received_at,
            label_ids=payload.get("labelIds", []),
        )


__all__ = ["GmailMessage", "GmailHistoryPage", "GmailProfile", "GmailAPIClient"]
