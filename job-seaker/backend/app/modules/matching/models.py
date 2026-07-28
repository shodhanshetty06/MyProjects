"""Matching module — ORM models.

Domain shape:
    MatchPreference — a user's structured job preferences (countries, work
                       mode, employment type, salary, experience). Owned by
                       this module rather than `user_profile` (still an
                       unimplemented Phase 0 stub) because rule-based
                       boosting and background batch matching both need a
                       persisted place to read preferences from outside of
                       a live request.
    ResumeJobMatch   — a computed match between one `ResumeVersion` and one
                        `Job`: semantic + skill + rule-boost scores, skill
                        gap, and narrative strengths/weaknesses. Unique per
                        (resume_version_id, job_id) — recomputing a pair
                        updates the existing row rather than duplicating it.
    Recommendation   — a generated, persisted suggestion (job, company, or
                        skill) for a user, derived from their best matches.

`user_id` columns have no foreign key, mirroring `resume_parser.Resume` —
the `auth`/`user_profile` modules are still unimplemented Phase 0 stubs, so
ownership is enforced at the service layer against the trusted `X-User-Id`
header (see `app/api/deps.py`).
"""

from __future__ import annotations

import enum
import uuid

from sqlalchemy import ForeignKey, Numeric, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base_class import Base, TimestampMixin, UUIDPrimaryKeyMixin


class MatchStatus(str, enum.Enum):
    PENDING = "pending"
    COMPUTED = "computed"
    STALE = "stale"
    FAILED = "failed"


class RecommendationType(str, enum.Enum):
    JOB = "job"
    COMPANY = "company"
    SKILL = "skill"


class RecommendationStatus(str, enum.Enum):
    ACTIVE = "active"
    DISMISSED = "dismissed"


class MatchPreference(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One user's structured preferences, used for rule-based match boosting
    and as the seed for background batch-matching/recommendation sweeps.
    """

    __tablename__ = "match_preference"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, unique=True, index=True
    )
    preferred_countries: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    preferred_work_modes: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    preferred_employment_types: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    years_of_experience: Mapped[int | None] = mapped_column(nullable=True)
    expected_salary_min: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    expected_salary_max: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    salary_currency: Mapped[str | None] = mapped_column(String(3), nullable=True)
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)


class ResumeJobMatch(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Computed match between one `ResumeVersion` and one `Job`."""

    __tablename__ = "resume_job_match"
    __table_args__ = (
        UniqueConstraint("resume_version_id", "job_id", name="uq_resume_job_match_pair"),
    )

    resume_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resume_version.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("job.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)

    status: Mapped[MatchStatus] = mapped_column(
        String(20), nullable=False, default=MatchStatus.PENDING
    )
    match_score: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    semantic_score: Mapped[float | None] = mapped_column(Numeric(6, 5), nullable=True)
    skill_match_score: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    rule_boost_score: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)

    matched_skills: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    missing_skills: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    strengths: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    weaknesses: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    score_breakdown: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # LLM-generated one-line "why this fits" summary — generated lazily on
    # first `GET .../why-score` request and cached here (see
    # `matching/narrative.py`), not computed for every match up front.
    ai_narrative: Mapped[str | None] = mapped_column(Text, nullable=True)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class Recommendation(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A generated, persisted recommendation (job/company/skill) for a user.

    Exactly one of `job_id` / `company_id` / `skill_id` is set, per
    `recommendation_type`. Refreshed in bulk (delete-then-insert per type,
    see `RecommendationRepository.replace_active`) rather than diffed in
    place — recommendation sets are cheap to regenerate wholesale and this
    avoids reconciling stale partial state.
    """

    __tablename__ = "recommendation"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    recommendation_type: Mapped[RecommendationType] = mapped_column(String(20), nullable=False)
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("job.id", ondelete="CASCADE"), nullable=True
    )
    company_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("company.id", ondelete="CASCADE"), nullable=True
    )
    skill_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("skill.id", ondelete="CASCADE"), nullable=True
    )
    score: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False, default=0)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[RecommendationStatus] = mapped_column(
        String(20), nullable=False, default=RecommendationStatus.ACTIVE
    )


__all__ = [
    "MatchStatus",
    "RecommendationType",
    "RecommendationStatus",
    "MatchPreference",
    "ResumeJobMatch",
    "Recommendation",
]
