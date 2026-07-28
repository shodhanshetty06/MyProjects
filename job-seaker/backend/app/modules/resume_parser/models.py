"""Resume Parser module — ORM models.

Two-level versioning:
    Resume         — the logical, named resume a user manages (title, owner).
    ResumeVersion  — one uploaded file. Exactly one version per resume is
                     `is_current` at a time (flipped inside a transaction by
                     the repository layer — see `repository.py`).

Each `ResumeVersion` owns at most one `ResumeParsedData` and one
`ResumeEmbedding` row, created once background parsing/embedding finishes.

`user_id` has no foreign key: the `auth` module is still an unimplemented
Phase 0 stub (no `User` table). Ownership is enforced at the service layer
against a trusted `X-User-Id` header — see `app/api/deps.py`.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base_class import Base, SoftDeleteMixin, TimestampMixin, UUIDPrimaryKeyMixin
from app.integrations.vector_store import VECTOR_DIMENSIONS


class ResumeVersionStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    PARSING = "parsing"
    PARSED = "parsed"
    FAILED = "failed"


class PipelineRunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ResumeContentType(str, enum.Enum):
    PDF = "pdf"
    DOCX = "docx"


class Resume(Base, UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin):
    """A logical, named resume owned by a user. Holds one or more versions."""

    __tablename__ = "resume"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)

    versions: Mapped[list["ResumeVersion"]] = relationship(
        back_populates="resume", cascade="all, delete-orphan", order_by="ResumeVersion.version_number"
    )


class ResumeVersion(Base, UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin):
    """A single uploaded resume file — one version of a `Resume`."""

    __tablename__ = "resume_version"
    __table_args__ = (UniqueConstraint("resume_id", "version_number", name="uq_resume_version_number"),)

    resume_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("resume.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    is_current: Mapped[bool] = mapped_column(nullable=False, default=False)
    status: Mapped[ResumeVersionStatus] = mapped_column(
        String(20), nullable=False, default=ResumeVersionStatus.UPLOADED
    )
    content_type: Mapped[ResumeContentType] = mapped_column(String(10), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    s3_bucket: Mapped[str] = mapped_column(String(255), nullable=False)
    s3_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- resume health (populated by the auto-orchestrated pipeline; see
    # `ResumePipelineRun` and `resume_parser.tasks.calculate_resume_score_task`
    # / `run_ats_baseline_analysis_task`) ---
    resume_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    resume_score_breakdown: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    ats_baseline_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ats_baseline_missing_skills: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    health_computed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    resume: Mapped["Resume"] = relationship(back_populates="versions")
    parsed_data: Mapped["ResumeParsedData | None"] = relationship(
        back_populates="version", cascade="all, delete-orphan", uselist=False
    )
    embedding: Mapped["ResumeEmbedding | None"] = relationship(
        back_populates="version", cascade="all, delete-orphan", uselist=False
    )


class ResumeParsedData(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Structured extraction output (Claude) for one `ResumeVersion`."""

    __tablename__ = "resume_parsed_data"

    resume_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resume_version.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    contact: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    skills: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    experience: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    education: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    projects: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    certifications: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    achievements: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    languages: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    model_used: Mapped[str] = mapped_column(String(100), nullable=False)

    version: Mapped["ResumeVersion"] = relationship(back_populates="parsed_data")


class ResumeEmbedding(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Vector embedding (Voyage AI) for one `ResumeVersion`, stored in pgvector."""

    __tablename__ = "resume_embedding"

    resume_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resume_version.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(VECTOR_DIMENSIONS), nullable=False)
    model_used: Mapped[str] = mapped_column(String(100), nullable=False)

    version: Mapped["ResumeVersion"] = relationship(back_populates="embedding")


class ResumePipelineRun(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Progress tracker for the auto-orchestrated resume pipeline.

    One row per `ResumeVersion` (recreated on re-upload/new version), updated
    in place by each task in the chain (parse -> embed -> score -> ATS
    baseline -> match -> recommend -> dashboard -> notify) so `GET
    /workspace/progress` has durable state to read on first paint/reconnect,
    with the Redis pub/sub channel (`app.modules.workspace.realtime`) layered
    on top for live push.
    """

    __tablename__ = "resume_pipeline_run"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    resume_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resume_version.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    status: Mapped[PipelineRunStatus] = mapped_column(
        String(20), nullable=False, default=PipelineRunStatus.PENDING
    )
    current_step: Mapped[str | None] = mapped_column(String(50), nullable=True)
    progress_pct: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    steps: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


__all__ = [
    "ResumeVersionStatus",
    "ResumeContentType",
    "PipelineRunStatus",
    "Resume",
    "ResumeVersion",
    "ResumeParsedData",
    "ResumeEmbedding",
    "ResumePipelineRun",
]
