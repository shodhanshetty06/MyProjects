"""SMTP email provider — the primary/default `EmailProvider` for development.
Works against any SMTP server (a local Mailpit/MailHog instance, a real
mailbox, etc.) with no cloud dependency.
"""

from __future__ import annotations

import asyncio
import smtplib
import uuid
from email.message import EmailMessage

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.notification.errors import NotificationSendError, TransientNotificationError

logger = get_logger(__name__)

_TRANSIENT_SMTP_ERRORS = (
    smtplib.SMTPConnectError,
    smtplib.SMTPServerDisconnected,
    smtplib.SMTPHeloError,
    ConnectionError,
    TimeoutError,
)


class SMTPEmailProvider:
    """Transactional email via `smtplib` (STARTTLS if `SMTP_USE_TLS`)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _send_sync(self, *, to: str, subject: str, body_text: str) -> str:
        message = EmailMessage()
        message["From"] = self._settings.SMTP_FROM_EMAIL
        message["To"] = to
        message["Subject"] = subject
        message_id = uuid.uuid4().hex
        message["Message-ID"] = f"<{message_id}@jobplatform.local>"
        message.set_content(body_text)

        try:
            with smtplib.SMTP(self._settings.SMTP_HOST, self._settings.SMTP_PORT, timeout=30) as client:
                if self._settings.SMTP_USE_TLS:
                    client.starttls()
                if self._settings.SMTP_USERNAME and self._settings.SMTP_PASSWORD:
                    client.login(self._settings.SMTP_USERNAME, self._settings.SMTP_PASSWORD)
                client.send_message(message)
        except _TRANSIENT_SMTP_ERRORS as exc:
            logger.error("SMTP send failed (transient)", extra={"extra_fields": {"error": str(exc)}})
            raise TransientNotificationError(str(exc)) from exc
        except smtplib.SMTPException as exc:
            logger.error("SMTP send failed", extra={"extra_fields": {"error": str(exc)}})
            raise NotificationSendError(str(exc)) from exc
        return message_id

    async def send(self, *, to: str, subject: str, body_text: str) -> str:
        return await asyncio.to_thread(self._send_sync, to=to, subject=subject, body_text=body_text)


__all__ = ["SMTPEmailProvider"]
