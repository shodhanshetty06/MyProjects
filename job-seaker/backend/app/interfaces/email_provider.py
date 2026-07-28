"""Contract every email provider (SMTP, SES, and future SendGrid/Mailgun
backends) must satisfy so callers never import boto3/smtplib directly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmailProvider(Protocol):
    """A transactional-email backend."""

    async def send(self, *, to: str, subject: str, body_text: str) -> str:
        """Send an email, returning a provider-assigned message id."""
        ...
