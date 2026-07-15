"""Failure Handling — the exception hierarchy adapters/tasks raise, and the
pure classifier that turns any exception into a `FailureCategory`.

`TransientApplyError` (and any `httpx`/Playwright timeout/network error) is
the only category Celery auto-retries — see `autoretry_for` on
`submit_application_task` in `tasks.py`, the same convention
`job_ingestion.fetch_company_jobs_task` uses for `httpx.HTTPError`.
Everything else needs a human to fix something (profile data, a CAPTCHA)
before a retry could possibly succeed, so it's surfaced instead of retried.
"""
from __future__ import annotations

import httpx

from app.core.exceptions import ServiceUnavailableError
from app.modules.auto_apply.models import FailureCategory


class AutoApplyError(Exception):
    """Base class for errors this module raises deliberately (as opposed to
    unexpected exceptions bubbling up from an adapter)."""

    category: FailureCategory = FailureCategory.UNKNOWN


class TransientApplyError(AutoApplyError):
    """A retryable failure — network blip, timeout, temporary 5xx, a page
    that didn't render in time."""

    category = FailureCategory.TRANSIENT


class CaptchaDetectedError(AutoApplyError):
    """The ATS presented a CAPTCHA/bot challenge — no amount of retrying
    will help without human intervention."""

    category = FailureCategory.CAPTCHA


class DuplicateApplicationError(AutoApplyError):
    """The ATS reports this applicant has already applied to this job."""

    category = FailureCategory.DUPLICATE


class PlatformApplyError(AutoApplyError):
    """The adapter hit something it doesn't know how to handle (an
    unexpected DOM/response shape) — worth a bounded number of retries in
    case it was a one-off rendering glitch, but likely needs an adapter fix
    if it keeps happening."""

    category = FailureCategory.PLATFORM_ERROR


_CAPTCHA_MARKERS = ("captcha", "are you a robot", "recaptcha", "hcaptcha", "cloudflare")


def classify_exception(exc: Exception) -> FailureCategory:
    """Map an arbitrary exception raised during a submission attempt to a
    `FailureCategory`, for attempts that didn't already raise one of the
    typed errors above.
    """
    if isinstance(exc, AutoApplyError):
        return exc.category

    message = str(exc).lower()
    if any(marker in message for marker in _CAPTCHA_MARKERS):
        return FailureCategory.CAPTCHA

    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return FailureCategory.TRANSIENT
    if isinstance(exc, ServiceUnavailableError):
        return FailureCategory.TRANSIENT
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500:
        return FailureCategory.TRANSIENT

    # Playwright's TimeoutError subclasses Exception directly (no shared
    # base with httpx) — matched by name so this module doesn't need a hard
    # dependency on `playwright` just to classify errors.
    if type(exc).__name__ in ("TimeoutError", "PlaywrightTimeoutError"):
        return FailureCategory.TRANSIENT

    return FailureCategory.UNKNOWN


RETRYABLE_CATEGORIES = (FailureCategory.TRANSIENT, FailureCategory.PLATFORM_ERROR)


def is_retryable(category: FailureCategory) -> bool:
    return category in RETRYABLE_CATEGORIES


__all__ = [
    "AutoApplyError",
    "TransientApplyError",
    "CaptchaDetectedError",
    "DuplicateApplicationError",
    "PlatformApplyError",
    "classify_exception",
    "is_retryable",
]
