"""Email Tracking module — ORM models.

Domain shape:
    GmailAccount   — one connected Gmail mailbox per user (1:1 today — a
                     second `/gmail/connect` for the same user just upserts
                     by `email_address`), holding encrypted OAuth tokens and
                     the incremental-sync cursor (`history_id`).
    TrackedEmail   — one row per Gmail message that was fetched and run
                     through Claude classification + application matching.
                     Durable audit trail: `raw_metadata` keeps Claude's full
                     structured response so a low-confidence/unmatched
                     classification can be reviewed and debugged later.

`user_id` has no foreign key: the `auth` module is still an unimplemented
Phase 0 stub (no `User` table). Ownership is enforced at the service layer
against a trusted `X-User-Id` header — see `app/api/deps.py`, same
convention `tracking.Application.user_id` already uses.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base_class import Base, TimestampMixin, UUIDPrimaryKeyMixin


class GmailAccountStatus(str, enum.Enum):
    CONNECTED = "connected"
    NEEDS_REAUTH = "needs_reauth"
    DISCONNECTED = "disconnected"


class EmailClassification(str, enum.Enum):
    INTERVIEW = "interview"
    ASSESSMENT = "assessment"
    REJECTION = "rejection"
    OFFER = "offer"
    NOT_RELEVANT = "not_relevant"


class EmailMatchStatus(str, enum.Enum):
    MATCHED = "matched"
    NEEDS_REVIEW = "needs_review"
    UNMATCHED = "unmatched"
    IGNORED = "ignored"


class GmailAccount(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A connected Gmail mailbox, polled periodically for status-update emails."""

    __tablename__ = "gmail_account"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    email_address: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    status: Mapped[GmailAccountStatus] = mapped_column(
        String(20), nullable=False, default=GmailAccountStatus.CONNECTED
    )
    access_token_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    refresh_token_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    token_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    scopes: Mapped[str] = mapped_column(String(512), nullable=False)
    history_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_sync_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    tracked_emails: Mapped[list[TrackedEmail]] = relationship(
        back_populates="gmail_account", cascade="all, delete-orphan"
    )


class TrackedEmail(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One Gmail message that was fetched, classified, and (maybe) matched."""

    __tablename__ = "tracked_email"
    __table_args__ = (
        UniqueConstraint(
            "gmail_account_id", "gmail_message_id", name="uq_tracked_email_account_message"
        ),
    )

    gmail_account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("gmail_account.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    application_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("application.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    gmail_message_id: Mapped[str] = mapped_column(String(100), nullable=False)
    thread_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    sender: Mapped[str] = mapped_column(String(512), nullable=False)
    subject: Mapped[str] = mapped_column(String(1024), nullable=False)
    snippet: Mapped[str] = mapped_column(Text, nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    classification: Mapped[EmailClassification] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    match_status: Mapped[EmailMatchStatus] = mapped_column(String(20), nullable=False, index=True)
    matched_status_history_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("application_status_history.id", ondelete="SET NULL"),
        nullable=True,
    )
    raw_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    gmail_account: Mapped[GmailAccount] = relationship(back_populates="tracked_emails")


__all__ = [
    "GmailAccountStatus",
    "EmailClassification",
    "EmailMatchStatus",
    "GmailAccount",
    "TrackedEmail",
]
