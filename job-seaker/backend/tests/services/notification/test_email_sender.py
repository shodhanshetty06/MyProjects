"""Unit tests for `SESEmailSender` — a fake boto3-shaped client records
`send_email` calls, matching the "fake provider client" convention used by
`tests/modules/resume_parser/test_ai_service.py` for the Anthropic client.
No real AWS credentials or network access required.
"""

from __future__ import annotations

import pytest
from botocore.exceptions import ClientError

from app.services.notification.email_sender import SESEmailSender
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


def test_send_returns_message_id_and_uses_configured_sender() -> None:
    fake_client = _FakeSESClient()
    sender = SESEmailSender(client=fake_client)

    message_id = sender.send(to="user@example.com", subject="Hello", body_text="Body text")

    assert message_id == "ses-message-id-123"
    call = fake_client.calls[0]
    assert call["Destination"]["ToAddresses"] == ["user@example.com"]
    assert call["Message"]["Subject"]["Data"] == "Hello"
    assert call["Message"]["Body"]["Text"]["Data"] == "Body text"
    assert call["Source"] == sender.sender


def test_throttling_error_raises_transient_notification_error() -> None:
    fake_client = _FakeSESClient(error=_client_error("Throttling"))
    sender = SESEmailSender(client=fake_client)

    with pytest.raises(TransientNotificationError):
        sender.send(to="user@example.com", subject="Hi", body_text="Body")


def test_message_rejected_raises_non_retryable_error() -> None:
    fake_client = _FakeSESClient(error=_client_error("MessageRejected"))
    sender = SESEmailSender(client=fake_client)

    with pytest.raises(NotificationSendError) as exc_info:
        sender.send(to="user@example.com", subject="Hi", body_text="Body")
    assert not isinstance(exc_info.value, TransientNotificationError)
