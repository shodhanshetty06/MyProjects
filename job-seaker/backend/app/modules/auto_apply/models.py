"""Auto Apply module — ORM models.

Domain shape:
    ApplicantProfile   — 1:1 per-user form-fill data (contact info, work
                          authorization, EEO answers, and a growing bank of
                          free-form question -> answer pairs) used to
                          populate ATS application forms. Lives in this
                          module rather than `user_profile` because that
                          module is still an unimplemented Phase 0 stub.
    AutoApplyTask       — one row per "apply to this job for this user"
                          request — the unit of work placed on the Celery
                          queue. Denormalizes company_name/job_title/job_url
                          from the target `Job` at creation time (same
                          snapshot convention as `tracking.Application`) so
                          the record stays meaningful even if the job is
                          later closed/soft-deleted.
    AutoApplyAttempt    — one row per execution attempt of a task (initial
                          run + every retry), mirroring `IngestionRun`'s
                          one-row-per-crawl-attempt convention.
    AutoApplyScreenshot — a captured page screenshot for one attempt,
                          stored in S3.
    AutoApplyLogEntry   — one immutable structured log line for one attempt,
                          the same append-only pattern as
                          `tracking.ApplicationNote`.

`user_id` has no foreign key: the `auth` module is still an unimplemented
Phase 0 stub (no `User` table). Ownership is enforced at the service layer
against a trusted `X-User-Id` header — see `app/api/deps.py`.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base_class import Base, SoftDeleteMixin, TimestampMixin, UUIDPrimaryKeyMixin
from app.modules.job_ingestion.models import ATSPlatform


class AutoApplyTaskStatus(str, enum.Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUBMITTED = "submitted"
    REQUIRES_REVIEW = "requires_review"
    RETRYING = "retrying"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AutoApplyAttemptStatus(str, enum.Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class FailureCategory(str, enum.Enum):
    TRANSIENT = "transient"
    VALIDATION = "validation"
    CAPTCHA = "captcha"
    DUPLICATE = "duplicate"
    PLATFORM_ERROR = "platform_error"
    UNKNOWN = "unknown"


class ScreenshotStage(str, enum.Enum):
    BEFORE_SUBMIT = "before_submit"
    AFTER_SUBMIT = "after_submit"
    MISSING_FIELDS = "missing_fields"
    STEP = "step"
    ERROR = "error"


class LogLevel(str, enum.Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ApplicantProfile(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """1:1 per-user data bank used to fill out ATS application forms."""

    __tablename__ = "applicant_profile"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, unique=True, index=True
    )
    auto_apply_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    first_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    last_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)

    address_line1: Mapped[str | None] = mapped_column(String(255), nullable=True)
    city: Mapped[str | None] = mapped_column(String(255), nullable=True)
    state: Mapped[str | None] = mapped_column(String(255), nullable=True)
    postal_code: Mapped[str | None] = mapped_column(String(20), nullable=True)
    country: Mapped[str | None] = mapped_column(String(255), nullable=True)

    linkedin_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    portfolio_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    github_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    website_url: Mapped[str | None] = mapped_column(String(512), nullable=True)

    work_authorized: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    requires_sponsorship: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    willing_to_relocate: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    years_of_experience: Mapped[int | None] = mapped_column(Integer, nullable=True)
    desired_salary: Mapped[str | None] = mapped_column(String(100), nullable=True)
    notice_period: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # EEO self-identification — all optional, free-text ("decline_to_answer" allowed).
    gender: Mapped[str | None] = mapped_column(String(100), nullable=True)
    race_ethnicity: Mapped[str | None] = mapped_column(String(100), nullable=True)
    veteran_status: Mapped[str | None] = mapped_column(String(100), nullable=True)
    disability_status: Mapped[str | None] = mapped_column(String(100), nullable=True)

    default_ats_resume_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ats_resume.id", ondelete="SET NULL"), nullable=True
    )
    default_resume_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("resume.id", ondelete="SET NULL"), nullable=True
    )

    # Growing bank of normalized-question -> answer pairs, used by
    # `field_mapping.py` to resolve custom screening questions that have no
    # dedicated column above (e.g. "Why do you want to work here?").
    custom_answers: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)


class AutoApplyTask(Base, UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin):
    """One "apply to this job for this user" request — a unit of queue work."""

    __tablename__ = "auto_apply_task"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("job.id", ondelete="SET NULL"), nullable=True, index=True
    )
    company_name: Mapped[str] = mapped_column(String(255), nullable=False)
    job_title: Mapped[str] = mapped_column(String(512), nullable=False)
    job_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    platform: Mapped[ATSPlatform] = mapped_column(String(20), nullable=False)

    # Explicit resume overrides (either may be null — resolved at run time by
    # `resume_selection.py` if unset).
    ats_resume_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ats_resume.id", ondelete="SET NULL"), nullable=True
    )
    resume_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("resume_version.id", ondelete="SET NULL"), nullable=True
    )
    resume_used_display_name: Mapped[str | None] = mapped_column(String(512), nullable=True)

    status: Mapped[AutoApplyTaskStatus] = mapped_column(
        String(20), nullable=False, default=AutoApplyTaskStatus.PENDING, index=True
    )
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)

    field_mapping_snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    missing_fields: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    confirmation_detected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    confirmation_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    confirmation_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    tracking_application_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("application.id", ondelete="SET NULL"), nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    attempts: Mapped[list["AutoApplyAttempt"]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="AutoApplyAttempt.attempt_number",
    )


class AutoApplyAttempt(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One execution attempt (initial run or retry) of an `AutoApplyTask`."""

    __tablename__ = "auto_apply_attempt"

    task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("auto_apply_task.id", ondelete="CASCADE"), nullable=False, index=True
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[AutoApplyAttemptStatus] = mapped_column(
        String(20), nullable=False, default=AutoApplyAttemptStatus.RUNNING
    )
    failure_category: Mapped[FailureCategory | None] = mapped_column(String(20), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    task: Mapped["AutoApplyTask"] = relationship(back_populates="attempts")
    screenshots: Mapped[list["AutoApplyScreenshot"]] = relationship(
        back_populates="attempt", cascade="all, delete-orphan"
    )
    logs: Mapped[list["AutoApplyLogEntry"]] = relationship(
        back_populates="attempt", cascade="all, delete-orphan", order_by="AutoApplyLogEntry.created_at"
    )


class AutoApplyScreenshot(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One captured page screenshot for an `AutoApplyAttempt`, stored in S3."""

    __tablename__ = "auto_apply_screenshot"

    attempt_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("auto_apply_attempt.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stage: Mapped[ScreenshotStage] = mapped_column(String(20), nullable=False)
    s3_bucket: Mapped[str] = mapped_column(String(255), nullable=False)
    s3_key: Mapped[str] = mapped_column(String(1024), nullable=False)

    attempt: Mapped["AutoApplyAttempt"] = relationship(back_populates="screenshots")


class AutoApplyLogEntry(Base, UUIDPrimaryKeyMixin):
    """One immutable structured log line for an `AutoApplyAttempt`."""

    __tablename__ = "auto_apply_log_entry"

    attempt_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("auto_apply_attempt.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    level: Mapped[LogLevel] = mapped_column(String(10), nullable=False, default=LogLevel.INFO)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    attempt: Mapped["AutoApplyAttempt"] = relationship(back_populates="logs")


__all__ = [
    "AutoApplyTaskStatus",
    "AutoApplyAttemptStatus",
    "FailureCategory",
    "ScreenshotStage",
    "LogLevel",
    "ApplicantProfile",
    "AutoApplyTask",
    "AutoApplyAttempt",
    "AutoApplyScreenshot",
    "AutoApplyLogEntry",
]
