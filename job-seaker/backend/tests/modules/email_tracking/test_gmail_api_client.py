"""Unit tests for `GmailAPIClient` — mocked `httpx` transport, no network
access required. Covers the `history.list` -> expired-cursor fallback path.
"""

from __future__ import annotations

import httpx
import pytest

from app.modules.email_tracking.errors import GmailHistoryExpiredError
from app.modules.email_tracking.gmail_api_client import GmailAPIClient


def _patch_client(monkeypatch: pytest.MonkeyPatch, client: GmailAPIClient, handler) -> None:
    monkeypatch.setattr(
        client, "_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


@pytest.mark.asyncio
async def test_get_profile_returns_email_and_history_id(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/profile")
        return httpx.Response(200, json={"emailAddress": "me@example.com", "historyId": "1000"})

    client = GmailAPIClient(access_token="token")
    _patch_client(monkeypatch, client, handler)

    profile = await client.get_profile()

    assert profile.email_address == "me@example.com"
    assert profile.history_id == "1000"


@pytest.mark.asyncio
async def test_list_history_collects_message_ids_across_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = [
        {
            "history": [{"messagesAdded": [{"message": {"id": "m1"}}]}],
            "nextPageToken": "page2",
            "historyId": "1001",
        },
        {
            "history": [{"messagesAdded": [{"message": {"id": "m2"}}]}],
            "historyId": "1002",
        },
    ]
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        response = pages[call_count["n"]]
        call_count["n"] += 1
        return httpx.Response(200, json=response)

    client = GmailAPIClient(access_token="token")
    _patch_client(monkeypatch, client, handler)

    page = await client.list_history(start_history_id="999", max_results=50)

    assert page.message_ids == ["m1", "m2"]
    assert page.next_history_id == "1002"


@pytest.mark.asyncio
async def test_list_history_raises_expired_error_on_404(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "historyId too old"}})

    client = GmailAPIClient(access_token="token")
    _patch_client(monkeypatch, client, handler)

    with pytest.raises(GmailHistoryExpiredError):
        await client.list_history(start_history_id="1", max_results=50)


@pytest.mark.asyncio
async def test_list_message_ids_returns_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["q"] == "after:100"
        return httpx.Response(200, json={"messages": [{"id": "a"}, {"id": "b"}]})

    client = GmailAPIClient(access_token="token")
    _patch_client(monkeypatch, client, handler)

    message_ids = await client.list_message_ids(query="after:100", max_results=10)

    assert message_ids == ["a", "b"]


@pytest.mark.asyncio
async def test_get_message_parses_headers_and_internal_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg-1",
                "threadId": "thread-1",
                "snippet": "We'd like to schedule an interview.",
                "internalDate": "1700000000000",
                "labelIds": ["INBOX"],
                "payload": {
                    "headers": [
                        {"name": "From", "value": "recruiting@acme.com"},
                        {"name": "Subject", "value": "Interview invite"},
                    ]
                },
            },
        )

    client = GmailAPIClient(access_token="token")
    _patch_client(monkeypatch, client, handler)

    message = await client.get_message(message_id="msg-1")

    assert message.sender == "recruiting@acme.com"
    assert message.subject == "Interview invite"
    assert message.snippet == "We'd like to schedule an interview."
    assert message.received_at.year == 2023
