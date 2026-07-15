"""Admin module — ORM models.

Domain shape:
    AdminActor           — a staff/admin account that operates the Admin
                            Portal (distinct from platform end-users).
    PlatformUserStatus    — 1:1 admin-owned overlay per platform `user_id`
                            (active/suspended/banned). There is no `User`
                            table anywhere in the platform yet (`auth` and
                            `user_profile` are still unimplemented Phase 0
                            stubs), so this keyed-by-bare-UUID overlay is the
                            only place user account status lives; "Users"
                            list/detail views aggregate activity from other
                            modules' `user_id` columns at read time.
    CompanyVerification   — 1:1 overlay per `job_ingestion.Company`.
    JobModeration         — 1:1 overlay per `job_ingestion.Job`.
    ContentReport         — a user-filed report against a job/company/user,
                            the shared queue behind "Reports" and "Content
                            Moderation".
    AuditLog              — one immutable row per admin mutation.
    FeatureFlag           — a togglable platform feature flag.

`user_id`/`reporter_user_id` carry no foreign key: same bare-UUID,
trusted-`X-User-Id`-header convention every other module uses (see
`app/api/deps.py`). `company_id`/`job_id` *do* carry real foreign keys since
`job_ingestion`'s tables already exist in the same database — the same
cross-module FK convention `tracking.Application.job_id` uses.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base_class import Base, TimestampMixin, UUIDPrimaryKeyMixin


class AdminRole(str, enum.Enum):
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    MODERATOR = "moderator"
    SUPPORT = "support"
    ANALYST = "analyst"


class UserAccountStatus(str, enum.Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    BANNED = "banned"


class VerificationStatus(str, enum.Enum):
    UNVERIFIED = "unverified"
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"


class ModerationStatus(str, enum.Enum):
    ACTIVE = "active"
    FLAGGED = "flagged"
    REMOVED = "removed"
    PENDING_REVIEW = "pending_review"


class ReportTargetType(str, enum.Enum):
    JOB = "job"
    COMPANY = "company"
    USER = "user"


class ReportStatus(str, enum.Enum):
    OPEN = "open"
    REVIEWING = "reviewing"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class AdminActor(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A staff account operating the Admin Portal."""

    __tablename__ = "admin_actor"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, unique=True, index=True
    )
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[AdminRole] = mapped_column(String(20), nullable=False, default=AdminRole.SUPPORT)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_active_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PlatformUserStatus(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Admin-owned account-status overlay for one platform `user_id`."""

    __tablename__ = "platform_user_status"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, unique=True, index=True
    )
    status: Mapped[UserAccountStatus] = mapped_column(
        String(20), nullable=False, default=UserAccountStatus.ACTIVE
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    changed_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("admin_actor.id", ondelete="SET NULL"), nullable=True
    )
    changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class CompanyVerification(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """1:1 verification overlay for one `job_ingestion.Company`."""

    __tablename__ = "company_verification"

    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("company.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    status: Mapped[VerificationStatus] = mapped_column(
        String(20), nullable=False, default=VerificationStatus.UNVERIFIED
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("admin_actor.id", ondelete="SET NULL"), nullable=True
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class JobModeration(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """1:1 moderation overlay for one `job_ingestion.Job`."""

    __tablename__ = "job_moderation"

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("job.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    status: Mapped[ModerationStatus] = mapped_column(
        String(20), nullable=False, default=ModerationStatus.ACTIVE
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    moderated_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("admin_actor.id", ondelete="SET NULL"), nullable=True
    )
    moderated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    flag_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class ContentReport(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A user-filed report against a job/company/user."""

    __tablename__ = "content_report"

    reporter_user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    target_type: Mapped[ReportTargetType] = mapped_column(String(20), nullable=False)
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[ReportStatus] = mapped_column(
        String(20), nullable=False, default=ReportStatus.OPEN, index=True
    )
    resolved_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("admin_actor.id", ondelete="SET NULL"), nullable=True
    )
    resolution_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuditLog(Base, UUIDPrimaryKeyMixin):
    """One immutable row per admin mutation — append-only, no `updated_at`."""

    __tablename__ = "audit_log"

    actor_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("admin_actor.id", ondelete="SET NULL"), nullable=True, index=True
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    target_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    target_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)
    before: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    after: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    actor: Mapped[AdminActor | None] = relationship()


class FeatureFlag(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A togglable platform feature flag."""

    __tablename__ = "feature_flag"

    key: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rollout_percentage: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    updated_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("admin_actor.id", ondelete="SET NULL"), nullable=True
    )


__all__ = [
    "AdminRole",
    "UserAccountStatus",
    "VerificationStatus",
    "ModerationStatus",
    "ReportTargetType",
    "ReportStatus",
    "AdminActor",
    "PlatformUserStatus",
    "CompanyVerification",
    "JobModeration",
    "ContentReport",
    "AuditLog",
    "FeatureFlag",
]
