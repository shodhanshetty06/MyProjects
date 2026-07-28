"""Job Ingestion module — ORM models.

Domain shape:
    Company              — an employer discovered/registered for ingestion.
    CompanyIntelligence  — 1:1 metadata about *how* to crawl a company (ATS
                            platform, career URL, adapter config) plus
                            crawl/observability fields ("Company Intelligence"
                            store: hiring frequency, last crawl, rating and
                            hiring-trend placeholders for future phases).
    Location             — normalized, deduplicated location, get-or-created
                            during job normalization.
    Skill                — normalized, deduplicated skill, get-or-created
                            during skill extraction. Linked to `Job` via the
                            `job_skill` association table.
    Job                  — a single job posting ingested from one ATS
                            adapter. Deduplicated on `(source, source_job_id)`;
                            `content_hash` drives change detection so
                            unchanged postings skip re-normalization.
    JobEmbedding          — 1:1 vector embedding for a `Job`, mirrors
                            `resume_parser.ResumeEmbedding`.
    IngestionRun          — one row per crawl attempt (scheduled or manual),
                            the basis for "last crawl" + retry observability.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Column,
    Computed,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base_class import Base, SoftDeleteMixin, TimestampMixin, UUIDPrimaryKeyMixin
from app.integrations.vector_store import VECTOR_DIMENSIONS


class ATSPlatform(str, enum.Enum):
    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    WORKDAY = "workday"
    ASHBY = "ashby"
    WORKABLE = "workable"
    SMARTRECRUITERS = "smartrecruiters"
    JOBVITE = "jobvite"
    CUSTOM_HTML = "custom_html"
    UNKNOWN = "unknown"


class SeniorityLevel(str, enum.Enum):
    """AI-inferred seniority band — display only, never fabricated without
    supporting text (see `SKILL_EXTRACTION_SYSTEM_PROMPT` -> job understanding
    prompt in `tasks.py`, which requires the model ground every field in the
    posting text or fall back to `unknown`)."""

    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    STAFF = "staff"
    PRINCIPAL = "principal"
    EXECUTIVE = "executive"
    UNKNOWN = "unknown"


class RemotePolicy(str, enum.Enum):
    """Company-level remote posture — distinct from `Job.work_mode`, which is
    per-posting. Set manually via `CompanyIntelligenceUpsertRequest` or
    inferred later from the mix of a company's job postings."""

    REMOTE_FIRST = "remote_first"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    FLEXIBLE = "flexible"
    UNKNOWN = "unknown"


class EmploymentType(str, enum.Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    TEMPORARY = "temporary"
    FREELANCE = "freelance"
    UNKNOWN = "unknown"


class WorkMode(str, enum.Enum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNKNOWN = "unknown"


class SalaryPeriod(str, enum.Enum):
    YEARLY = "yearly"
    MONTHLY = "monthly"
    HOURLY = "hourly"
    UNKNOWN = "unknown"


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    NORMALIZING = "normalizing"
    ACTIVE = "active"
    FAILED = "failed"
    CLOSED = "closed"


class IngestionRunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class Company(Base, UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin):
    """An employer discovered or registered for job ingestion."""

    __tablename__ = "company"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_name: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    website: Mapped[str | None] = mapped_column(String(512), nullable=True)
    logo_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    industry: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    intelligence: Mapped[CompanyIntelligence | None] = relationship(
        back_populates="company", cascade="all, delete-orphan", uselist=False
    )
    jobs: Mapped[list[Job]] = relationship(back_populates="company", cascade="all, delete-orphan")
    ingestion_runs: Mapped[list[IngestionRun]] = relationship(
        back_populates="company", cascade="all, delete-orphan"
    )


class CompanyIntelligence(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Crawl configuration + observability metadata for one `Company`."""

    __tablename__ = "company_intelligence"

    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("company.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    ats_platform: Mapped[ATSPlatform] = mapped_column(
        String(20), nullable=False, default=ATSPlatform.UNKNOWN
    )
    ats_identifier: Mapped[str | None] = mapped_column(String(255), nullable=True)
    career_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    adapter_config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    last_crawled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_crawl_status: Mapped[IngestionRunStatus | None] = mapped_column(String(20), nullable=True)
    hiring_frequency_per_month: Mapped[float | None] = mapped_column(Numeric(10, 2), nullable=True)
    # Glassdoor-style aggregate rating (0.00-5.00). Named generically since the
    # source may not literally be Glassdoor.
    company_rating: Mapped[float | None] = mapped_column(Numeric(3, 2), nullable=True)
    hiring_trends: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # --- Company Intelligence profile fields (Feature 4) ---
    company_size: Mapped[str | None] = mapped_column(String(50), nullable=True)
    funding_stage: Mapped[str | None] = mapped_column(String(50), nullable=True)
    remote_policy: Mapped[RemotePolicy] = mapped_column(
        String(20), nullable=False, default=RemotePolicy.UNKNOWN
    )
    tech_stack: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)

    company: Mapped[Company] = relationship(back_populates="intelligence")


class Location(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A normalized, deduplicated job location."""

    __tablename__ = "location"

    raw_text: Mapped[str] = mapped_column(String(512), nullable=False)
    city: Mapped[str | None] = mapped_column(String(255), nullable=True)
    region: Mapped[str | None] = mapped_column(String(255), nullable=True)
    country: Mapped[str | None] = mapped_column(String(255), nullable=True)
    country_code: Mapped[str | None] = mapped_column(String(2), nullable=True)
    normalized_key: Mapped[str] = mapped_column(
        String(512), nullable=False, unique=True, index=True
    )

    jobs: Mapped[list[Job]] = relationship(back_populates="location")


class Skill(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A normalized, deduplicated skill extracted from job descriptions."""

    __tablename__ = "skill"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_name: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )

    jobs: Mapped[list[Job]] = relationship(secondary="job_skill", back_populates="skills")


job_skill = Table(
    "job_skill",
    Base.metadata,
    Column(
        "job_id", UUID(as_uuid=True), ForeignKey("job.id", ondelete="CASCADE"), primary_key=True
    ),
    Column(
        "skill_id", UUID(as_uuid=True), ForeignKey("skill.id", ondelete="CASCADE"), primary_key=True
    ),
)


class Job(Base, UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin):
    """A single job posting ingested from one ATS adapter."""

    __tablename__ = "job"
    __table_args__ = (
        UniqueConstraint("source", "source_job_id", name="uq_job_source_source_job_id"),
    )

    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("company.id", ondelete="CASCADE"), nullable=False, index=True
    )
    location_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("location.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    source: Mapped[ATSPlatform] = mapped_column(String(20), nullable=False)
    source_job_id: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    employment_type: Mapped[EmploymentType] = mapped_column(
        String(20), nullable=False, default=EmploymentType.UNKNOWN
    )
    work_mode: Mapped[WorkMode] = mapped_column(
        String(20), nullable=False, default=WorkMode.UNKNOWN
    )
    salary_min: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    salary_max: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    salary_currency: Mapped[str | None] = mapped_column(String(3), nullable=True)
    salary_period: Mapped[SalaryPeriod] = mapped_column(
        String(20), nullable=False, default=SalaryPeriod.UNKNOWN
    )
    salary_raw_text: Mapped[str | None] = mapped_column(String(255), nullable=True)
    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    status: Mapped[JobStatus] = mapped_column(String(20), nullable=False, default=JobStatus.PENDING)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    raw_payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # --- AI Job Understanding (Feature 3) — populated by the "enrich" stage
    # in tasks.py, grounded in the posting text (see the system prompt).
    # `skills` (the Skill relationship below) remains the union of
    # required+preferred for search/matching continuity; these columns keep
    # the finer-grained breakdown for display and future scoring. ---
    required_skills: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    preferred_skills: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    responsibilities: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    benefits: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    tech_stack: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    seniority: Mapped[SeniorityLevel] = mapped_column(
        String(20), nullable=False, default=SeniorityLevel.UNKNOWN
    )
    industry: Mapped[str | None] = mapped_column(String(255), nullable=True)
    department: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # --- Job Quality Score (Feature 5) — deterministic, rule-based (see
    # `quality.py`); recomputed whenever the posting's content changes so it
    # never drifts from the current data (no learned/trained weights, so
    # there's nothing here that can overfit to historical data). ---
    quality_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quality_reasons: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    # Postgres-generated `GENERATED ALWAYS AS (...) STORED` column (title
    # weighted over description) — the DDL itself is created by the search
    # module's `0003_search_indexes` migration; `Computed(...)` here just
    # tells the ORM this column is database-managed (never included in
    # INSERT/UPDATE) so `SearchRepository` can reference it in `@@`/rank
    # queries. `deferred=True` keeps it out of normal `Job` loads.
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', coalesce(title, '')), 'A') || "
            "setweight(to_tsvector('english', coalesce(description, '')), 'B')",
            persisted=True,
        ),
        nullable=True,
        deferred=True,
    )

    company: Mapped[Company] = relationship(back_populates="jobs")
    location: Mapped[Location | None] = relationship(back_populates="jobs")
    skills: Mapped[list[Skill]] = relationship(secondary="job_skill", back_populates="jobs")
    embedding: Mapped[JobEmbedding | None] = relationship(
        back_populates="job", cascade="all, delete-orphan", uselist=False
    )


class JobEmbedding(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Vector embedding (Voyage AI) for one `Job`, stored in pgvector."""

    __tablename__ = "job_embedding"

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("job.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(VECTOR_DIMENSIONS), nullable=False)
    model_used: Mapped[str] = mapped_column(String(100), nullable=False)

    job: Mapped[Job] = relationship(back_populates="embedding")


class IngestionRun(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One crawl attempt (scheduled sweep or manual trigger) for a `Company`."""

    __tablename__ = "ingestion_run"

    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("company.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source: Mapped[ATSPlatform] = mapped_column(String(20), nullable=False)
    status: Mapped[IngestionRunStatus] = mapped_column(
        String(20), nullable=False, default=IngestionRunStatus.PENDING
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    jobs_found: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    jobs_created: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    jobs_updated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    jobs_closed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    jobs_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    company: Mapped[Company] = relationship(back_populates="ingestion_runs")


__all__ = [
    "ATSPlatform",
    "EmploymentType",
    "WorkMode",
    "SalaryPeriod",
    "JobStatus",
    "IngestionRunStatus",
    "SeniorityLevel",
    "RemotePolicy",
    "Company",
    "CompanyIntelligence",
    "Location",
    "Skill",
    "job_skill",
    "Job",
    "JobEmbedding",
    "IngestionRun",
]
