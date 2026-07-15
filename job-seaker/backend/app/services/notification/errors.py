"""Notification channel-provider error hierarchy + retry classification.

Same shape as `app/modules/auto_apply/errors.py` and
`app/modules/email_tracking/errors.py`: a typed exception with a `category`,
and a pure classifier so callers (Celery tasks) can decide whether to retry.
"""

from __future__ import annotations

import enum


class NotificationFailureCategory(str, enum.Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"


class NotificationSendError(Exception):
    """Base class for channel-provider send failures."""

    category: NotificationFailureCategory = NotificationFailureCategory.PERMANENT


class TransientNotificationError(NotificationSendError):
    """A retryable failure — throttling, timeout, temporary 5xx."""

    category = NotificationFailureCategory.TRANSIENT


_TRANSIENT_SES_ERROR_CODES = frozenset({"Throttling", "ThrottlingException", "ServiceUnavailable"})


def classify_ses_exception(exc: Exception) -> NotificationFailureCategory:
    """Classify an exception raised while sending a notification email.

    `SESEmailSender.send` already converts a raw `ClientError` into
    `TransientNotificationError`/`NotificationSendError` before it reaches a
    Celery task's except block, so already-typed exceptions are classified
    via their `category` attribute first — same pattern as
    `auto_apply/errors.py::classify_exception`. Only a *raw*
    `botocore.exceptions.ClientError` (matched by its `.response` attribute
    rather than an isinstance check, so this module doesn't need a hard
    `botocore` import) falls through to the AWS-error-code check below.
    """
    if isinstance(exc, NotificationSendError):
        return exc.category

    response = getattr(exc, "response", None)
    error_code = response.get("Error", {}).get("Code") if isinstance(response, dict) else None
    if error_code in _TRANSIENT_SES_ERROR_CODES:
        return NotificationFailureCategory.TRANSIENT
    return NotificationFailureCategory.PERMANENT


def is_retryable(category: NotificationFailureCategory) -> bool:
    return category == NotificationFailureCategory.TRANSIENT


__all__ = [
    "NotificationFailureCategory",
    "NotificationSendError",
    "TransientNotificationError",
    "classify_ses_exception",
    "is_retryable",
]
