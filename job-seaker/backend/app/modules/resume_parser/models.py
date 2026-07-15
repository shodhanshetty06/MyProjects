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

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base_class import Base, SoftDeleteMixin, TimestampMixin, UUIDPrimaryKeyMixin
from app.integrations.vector_store import VECTOR_DIMENSIONS


class ResumeVersionStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    PARSING = "parsing"
    PARSED = "parsed"
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


__all__ = [
    "ResumeVersionStatus",
    "ResumeContentType",
    "Resume",
    "ResumeVersion",
    "ResumeParsedData",
    "ResumeEmbedding",
]
