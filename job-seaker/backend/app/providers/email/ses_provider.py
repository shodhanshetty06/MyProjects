"""AWS SES email provider — ported from the pre-refactor `SESEmailSender`.
Ready to select via `EMAIL_PROVIDER=ses`; no other application code changes
needed.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from app.core.config import Settings
from app.services.notification.errors import (
    NotificationFailureCategory,
    NotificationSendError,
    TransientNotificationError,
    classify_ses_exception,
)


@lru_cache
def _get_ses_client(region: str, access_key: str | None, secret_key: str | None) -> BaseClient:
    return boto3.client(
        "ses", region_name=region, aws_access_key_id=access_key, aws_secret_access_key=secret_key
    )


class SESEmailProvider:
    """Transactional email via AWS SES `send_email`."""

    def __init__(self, settings: Settings, client: BaseClient | None = None) -> None:
        self.client = client or _get_ses_client(
            settings.AWS_REGION, settings.AWS_ACCESS_KEY_ID, settings.AWS_SECRET_ACCESS_KEY
        )
        self.sender = settings.NOTIFICATION_SES_SENDER_EMAIL
        self.configuration_set = settings.NOTIFICATION_SES_CONFIGURATION_SET

    def _send_sync(self, *, to: str, subject: str, body_text: str) -> str:
        kwargs: dict = {
            "Source": self.sender,
            "Destination": {"ToAddresses": [to]},
            "Message": {
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {"Text": {"Data": body_text, "Charset": "UTF-8"}},
            },
        }
        if self.configuration_set:
            kwargs["ConfigurationSetName"] = self.configuration_set

        try:
            response = self.client.send_email(**kwargs)
        except ClientError as exc:
            category = classify_ses_exception(exc)
            if category == NotificationFailureCategory.TRANSIENT:
                raise TransientNotificationError(str(exc)) from exc
            raise NotificationSendError(str(exc)) from exc
        return response["MessageId"]

    async def send(self, *, to: str, subject: str, body_text: str) -> str:
        return await asyncio.to_thread(self._send_sync, to=to, subject=subject, body_text=body_text)


__all__ = ["SESEmailProvider"]
