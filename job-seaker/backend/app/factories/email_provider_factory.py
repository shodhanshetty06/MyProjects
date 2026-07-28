"""Picks the concrete `EmailProvider` implementation from
`settings.EMAIL_PROVIDER`.

`sendgrid`/`mailgun` are recognized as valid future values (per the spec's
Provider Pattern goal) but raise `NotImplementedError` until a concrete
`providers/email/*` class is added for them — no config schema for those
was specified, unlike `smtp`/`ses`.
"""

from __future__ import annotations

from app.core.config import Settings
from app.interfaces.email_provider import EmailProvider

_FUTURE_PROVIDERS = {"sendgrid", "mailgun"}


def build_email_provider(settings: Settings) -> EmailProvider:
    provider = settings.EMAIL_PROVIDER.lower()

    if provider == "smtp":
        from app.providers.email.smtp_provider import SMTPEmailProvider

        return SMTPEmailProvider(settings)
    if provider == "ses":
        from app.providers.email.ses_provider import SESEmailProvider

        return SESEmailProvider(settings)
    if provider in _FUTURE_PROVIDERS:
        raise NotImplementedError(
            f"EMAIL_PROVIDER '{provider}' is not implemented yet. "
            "Add a providers/email/{provider}_provider.py conforming to EmailProvider."
        )

    raise ValueError(f"Unknown EMAIL_PROVIDER '{settings.EMAIL_PROVIDER}'. Expected: smtp, ses.")


__all__ = ["build_email_provider"]
