"""AWS SES email sending — the pure, DB-free channel provider for the
notification module's EMAIL channel.

Wraps `boto3.client("ses", ...)` the same way `app/integrations/s3_client.py`
wraps the S3 client: a cached client factory plus a thin convenience class,
reusing the AWS credentials/region already configured for S3 (`AWS_ACCESS_KEY_ID`
/`AWS_SECRET_ACCESS_KEY`/`AWS_REGION` in `Settings`) rather than adding a new
provider/secret.
"""

from __future__ import annotations

from functools import lru_cache

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from app.core.config import get_settings
from app.services.notification.errors import (
    NotificationFailureCategory,
    NotificationSendError,
    TransientNotificationError,
    classify_ses_exception,
)


@lru_cache
def get_ses_client() -> BaseClient:
    settings = get_settings()
    return boto3.client(
        "ses",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )


class SESEmailSender:
    """Thin convenience wrapper around SES `send_email`."""

    def __init__(self, client: BaseClient | None = None) -> None:
        settings = get_settings()
        self.client = client or get_ses_client()
        self.sender = settings.NOTIFICATION_SES_SENDER_EMAIL
        self.configuration_set = settings.NOTIFICATION_SES_CONFIGURATION_SET

    def send(self, *, to: str, subject: str, body_text: str) -> str:
        """Synchronous SES call — callers running on an event loop should
        offload via `asyncio.to_thread`, the same pattern `AIService` uses
        for the synchronous Voyage SDK."""
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


__all__ = ["SESEmailSender", "get_ses_client"]
