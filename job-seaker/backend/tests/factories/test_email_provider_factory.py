"""Provider-switching tests for `build_email_provider`."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.factories.email_provider_factory import build_email_provider
from app.providers.email.ses_provider import SESEmailProvider
from app.providers.email.smtp_provider import SMTPEmailProvider


def test_build_email_provider_selects_smtp() -> None:
    settings = Settings(EMAIL_PROVIDER="smtp")

    provider = build_email_provider(settings)

    assert isinstance(provider, SMTPEmailProvider)


def test_build_email_provider_selects_ses() -> None:
    settings = Settings(EMAIL_PROVIDER="ses")

    provider = build_email_provider(settings)

    assert isinstance(provider, SESEmailProvider)


@pytest.mark.parametrize("provider_name", ["sendgrid", "mailgun"])
def test_build_email_provider_future_providers_raise_not_implemented(provider_name) -> None:
    settings = Settings(EMAIL_PROVIDER=provider_name)

    with pytest.raises(NotImplementedError):
        build_email_provider(settings)


def test_build_email_provider_rejects_unknown_provider() -> None:
    settings = Settings(EMAIL_PROVIDER="not-a-real-provider")

    with pytest.raises(ValueError, match="Unknown EMAIL_PROVIDER"):
        build_email_provider(settings)
