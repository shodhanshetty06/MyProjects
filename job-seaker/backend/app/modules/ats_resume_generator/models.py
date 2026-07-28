"""ATS Resume Generator module — ORM models.

Domain shape:
    ATSResume        — a logical, named "tailored resume" a user manages,
                        derived from one of their `resume_parser.Resume`s
                        and (optionally) targeted at a `job_ingestion.Job`.
    ATSResumeVersion — one optimization pass. Exactly one version per
                        `ATSResume` is `is_current` at a time (same
                        set-current bookkeeping as `resume_parser.ResumeVersion`).

                        Only `summary`, `skills`, and `projects` are ever
                        sent to Claude for rewriting (Claude additionally
                        regroups `skills` into `grouped_skills`, a strict
                        repartition of the same list) — `contact`,
                        `experience`, `education`, `certifications`,
                        `languages`, and `achievements` are copied verbatim
                        from the source resume's parsed data and never
                        touched by the model, so the generator can never
                        fabricate work history or contact details.
                        `original_*` snapshots are kept alongside `optimized_*`
                        so a version can be diffed against its own source
                        without a second DB round-trip. `template_id`
                        selects the visual template used to render the
                        DOCX/PDF (see `templates.py`) — a pure rendering
                        choice, independent of optimization.
    ATSResumeFile    — one generated file artifact (DOCX or PDF) for a
                        version, stored in S3. At most one row per
                        `(version, format)`.

`user_id` has no foreign key, mirroring `resume_parser.Resume` — the `auth`
module is still an unimplemented Phase 0 stub, so ownership is enforced at
the service layer against the trusted `X-User-Id` header (see
`app/api/deps.py`).
"""
from __future__ import annotations

import enum
import uuid

from sqlalchemy import ForeignKey, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base_class import Base, SoftDeleteMixin, TimestampMixin, UUIDPrimaryKeyMixin


class ATSVersionStatus(str, enum.Enum):
    PENDING = "pending"
    OPTIMIZING = "optimizing"
    GENERATING_FILES = "generating_files"
    COMPLETED = "completed"
    FAILED = "failed"


class ATSFileFormat(str, enum.Enum):
    DOCX = "docx"
    PDF = "pdf"


class ATSResume(Base, UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin):
    """A logical, named tailored resume owned by a user. Holds one or more versions."""

    __tablename__ = "ats_resume"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    source_resume_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("resume.id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("job.id", ondelete="SET NULL"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    job_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    company_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    versions: Mapped[list["ATSResumeVersion"]] = relationship(
        back_populates="ats_resume",
        cascade="all, delete-orphan",
        order_by="ATSResumeVersion.version_number",
    )


class ATSResumeVersion(Base, UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin):
    """One optimization pass over an `ATSResume`."""

    __tablename__ = "ats_resume_version"
    __table_args__ = (
        UniqueConstraint("ats_resume_id", "version_number", name="uq_ats_resume_version_number"),
    )

    ats_resume_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ats_resume.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    is_current: Mapped[bool] = mapped_column(nullable=False, default=False)
    status: Mapped[ATSVersionStatus] = mapped_column(
        String(20), nullable=False, default=ATSVersionStatus.PENDING
    )
    source_resume_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("resume_version.id", ondelete="CASCADE"), nullable=False
    )
    job_description: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Verbatim sections copied from the source resume's parsed data — never
    # sent to or rewritten by Claude, so work history can't be fabricated.
    contact: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    experience: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    education: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    certifications: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    languages: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    achievements: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # Original snapshot (for diffing) + AI-optimized rewrite of the three
    # sections Claude is allowed to touch.
    original_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    optimized_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    original_skills: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    optimized_skills: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    original_projects: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    optimized_projects: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # AI-organized skill categories (e.g. [{"category": "Languages", "skills": [...]}])
    # — a strict regrouping of `optimized_skills`, never introducing new skills.
    grouped_skills: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # Selected visual template for DOCX/PDF rendering — see
    # `ats_resume_generator.templates.TEMPLATES`. Purely a rendering choice;
    # switching it re-triggers file generation only, never re-optimization.
    template_id: Mapped[str] = mapped_column(String(50), nullable=False, default="modern_ats")

    # Non-blocking issues from the pre-export validator (see `validation.py`),
    # e.g. a project with fewer than 3 bullet points. Populated during file
    # generation; an empty list means nothing to flag.
    validation_warnings: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    keywords: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    matched_keywords: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    missing_keywords: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    keyword_score: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    ats_score: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    score_breakdown: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    model_used: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    ats_resume: Mapped["ATSResume"] = relationship(back_populates="versions")
    files: Mapped[list["ATSResumeFile"]] = relationship(
        back_populates="version", cascade="all, delete-orphan"
    )


class ATSResumeFile(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One generated file artifact (DOCX or PDF) for an `ATSResumeVersion`."""

    __tablename__ = "ats_resume_file"
    __table_args__ = (
        UniqueConstraint("ats_resume_version_id", "format", name="uq_ats_resume_file_format"),
    )

    ats_resume_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ats_resume_version.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    format: Mapped[ATSFileFormat] = mapped_column(String(10), nullable=False)
    s3_bucket: Mapped[str] = mapped_column(String(255), nullable=False)
    s3_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    version: Mapped["ATSResumeVersion"] = relationship(back_populates="files")


__all__ = [
    "ATSVersionStatus",
    "ATSFileFormat",
    "ATSResume",
    "ATSResumeVersion",
    "ATSResumeFile",
]
