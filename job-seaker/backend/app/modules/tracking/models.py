"""Tracking module — ORM models.

Domain shape:
    Application               — one job application a user is tracking,
                                 optionally linked to a `job_ingestion.Job`
                                 (manually-tracked applications instead carry
                                 their own company_name/job_title/job_url).
    ApplicationStatusHistory   — one immutable row per status transition;
                                 the source of both "Status History" and,
                                 merged with notes, the application "Timeline".
    ApplicationNote            — a free-form, user-authored note attached to
                                 an application. Full CRUD.

`user_id` has no foreign key: the `auth` module is still an unimplemented
Phase 0 stub (no `User` table). Ownership is enforced at the service layer
against a trusted `X-User-Id` header — see `app/api/deps.py`.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base_class import Base, SoftDeleteMixin, TimestampMixin, UUIDPrimaryKeyMixin


class ApplicationStatus(str, enum.Enum):
    APPLIED = "applied"
    INTERVIEW = "interview"
    ASSESSMENT = "assessment"
    REJECTED = "rejected"
    OFFER = "offer"
    ACCEPTED = "accepted"
    WITHDRAWN = "withdrawn"


class ApplicationSource(str, enum.Enum):
    MANUAL = "manual"
    AUTO_APPLY = "auto_apply"


class Application(Base, UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin):
    """A job application tracked by a user, through to its final outcome."""

    __tablename__ = "application"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("job.id", ondelete="SET NULL"), nullable=True, index=True
    )
    company_name: Mapped[str] = mapped_column(String(255), nullable=False)
    job_title: Mapped[str] = mapped_column(String(512), nullable=False)
    job_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    location: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[ApplicationStatus] = mapped_column(
        String(20), nullable=False, default=ApplicationStatus.APPLIED, index=True
    )
    source: Mapped[ApplicationSource] = mapped_column(
        String(20), nullable=False, default=ApplicationSource.MANUAL
    )
    applied_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    status_history: Mapped[list["ApplicationStatusHistory"]] = relationship(
        back_populates="application",
        cascade="all, delete-orphan",
        order_by="ApplicationStatusHistory.created_at",
    )
    notes: Mapped[list["ApplicationNote"]] = relationship(
        back_populates="application",
        cascade="all, delete-orphan",
        order_by="ApplicationNote.created_at.desc()",
    )


class ApplicationStatusHistory(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One immutable row per status transition on an `Application`."""

    __tablename__ = "application_status_history"

    application_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("application.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    from_status: Mapped[ApplicationStatus | None] = mapped_column(String(20), nullable=True)
    to_status: Mapped[ApplicationStatus] = mapped_column(String(20), nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    application: Mapped["Application"] = relationship(back_populates="status_history")


class ApplicationNote(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A free-form, user-authored note attached to an `Application`."""

    __tablename__ = "application_note"

    application_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("application.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)

    application: Mapped["Application"] = relationship(back_populates="notes")


__all__ = [
    "ApplicationStatus",
    "ApplicationSource",
    "Application",
    "ApplicationStatusHistory",
    "ApplicationNote",
]
