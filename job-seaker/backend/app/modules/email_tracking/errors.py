"""Failure Handling — the exception hierarchy Gmail-integration code raises,
and the pure classifier that turns any exception into a `GmailFailureCategory`.

`TransientGmailError` (network/timeout/5xx/429) is the only category Celery
auto-retries — see `autoretry_for` on the tasks in `tasks.py`, the same
convention `auto_apply/errors.py::classify_exception` uses. `GmailAuthError`
(invalid_grant / 401 on token refresh) is deliberately NOT retried: no amount
of retrying fixes a revoked grant, so the account is instead flagged
`NEEDS_REAUTH` for the user to reconnect. Everything else is logged and the
single message/poll is skipped rather than failing the whole batch.
"""

from __future__ import annotations

import enum

import httpx


class GmailFailureCategory(str, enum.Enum):
    TRANSIENT = "transient"
    AUTH = "auth"
    PERMANENT = "permanent"
    UNKNOWN = "unknown"


class EmailTrackingError(Exception):
    """Base class for errors this module raises deliberately."""

    category: GmailFailureCategory = GmailFailureCategory.UNKNOWN


class TransientGmailError(EmailTrackingError):
    """A retryable failure — network blip, timeout, temporary 5xx, rate limit."""

    category = GmailFailureCategory.TRANSIENT


class GmailAuthError(EmailTrackingError):
    """The refresh token is invalid/revoked — needs the user to reconnect."""

    category = GmailFailureCategory.AUTH


class GmailHistoryExpiredError(EmailTrackingError):
    """Gmail's `history.list` 404s once `history_id` is too stale (~7 days).

    Not itself a retry signal — callers should catch this and fall back to
    `GmailAPIClient.list_message_ids`.
    """

    category = GmailFailureCategory.PERMANENT


def classify_gmail_exception(exc: Exception) -> GmailFailureCategory:
    if isinstance(exc, EmailTrackingError):
        return exc.category

    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return GmailFailureCategory.TRANSIENT
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        if status_code == 401:
            return GmailFailureCategory.AUTH
        if status_code == 429 or status_code >= 500:
            return GmailFailureCategory.TRANSIENT
        return GmailFailureCategory.PERMANENT

    return GmailFailureCategory.UNKNOWN


def is_retryable(category: GmailFailureCategory) -> bool:
    return category == GmailFailureCategory.TRANSIENT


__all__ = [
    "GmailFailureCategory",
    "EmailTrackingError",
    "TransientGmailError",
    "GmailAuthError",
    "GmailHistoryExpiredError",
    "classify_gmail_exception",
    "is_retryable",
]
