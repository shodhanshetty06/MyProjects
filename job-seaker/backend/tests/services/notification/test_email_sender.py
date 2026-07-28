"""Unit tests for `SESEmailProvider` — a fake boto3-shaped client records
`send_email` calls, matching the "fake provider client" convention used
elsewhere in the test suite. No real AWS credentials or network access
required.
"""

from __future__ import annotations

import pytest
from botocore.exceptions import ClientError

from app.core.config import get_settings
from app.providers.email.ses_provider import SESEmailProvider
from app.services.notification.errors import NotificationSendError, TransientNotificationError


class _FakeSESClient:
    def __init__(self, *, error: Exception | None = None) -> None:
        self._error = error
        self.calls: list[dict] = []

    def send_email(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return {"MessageId": "ses-message-id-123"}


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "boom"}}, "SendEmail")


@pytest.mark.asyncio
async def test_send_returns_message_id_and_uses_configured_sender() -> None:
    fake_client = _FakeSESClient()
    sender = SESEmailProvider(get_settings(), client=fake_client)

    message_id = await sender.send(to="user@example.com", subject="Hello", body_text="Body text")

    assert message_id == "ses-message-id-123"
    call = fake_client.calls[0]
    assert call["Destination"]["ToAddresses"] == ["user@example.com"]
    assert call["Message"]["Subject"]["Data"] == "Hello"
    assert call["Message"]["Body"]["Text"]["Data"] == "Body text"
    assert call["Source"] == sender.sender


@pytest.mark.asyncio
async def test_throttling_error_raises_transient_notification_error() -> None:
    fake_client = _FakeSESClient(error=_client_error("Throttling"))
    sender = SESEmailProvider(get_settings(), client=fake_client)

    with pytest.raises(TransientNotificationError):
        await sender.send(to="user@example.com", subject="Hi", body_text="Body")


@pytest.mark.asyncio
async def test_message_rejected_raises_non_retryable_error() -> None:
    fake_client = _FakeSESClient(error=_client_error("MessageRejected"))
    sender = SESEmailProvider(get_settings(), client=fake_client)

    with pytest.raises(NotificationSendError) as exc_info:
        await sender.send(to="user@example.com", subject="Hi", body_text="Body")
    assert not isinstance(exc_info.value, TransientNotificationError)
