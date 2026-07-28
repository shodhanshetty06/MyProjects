"""Matching module — data access layer.

Reads `resume_parser` and `job_ingestion` models directly (cross-module
reads over shared `Base.metadata`, the same convention `search` already uses
for `job_ingestion`) in addition to owning this module's own three tables.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.modules.job_ingestion.models import Company, Job, JobEmbedding, JobStatus, Skill
from app.modules.matching.models import (
    MatchPreference,
    MatchStatus,
    Recommendation,
    RecommendationStatus,
    RecommendationType,
    ResumeJobMatch,
)
from app.modules.resume_parser.models import Resume, ResumeVersion
from app.repositories.base import BaseRepository

_JOB_EAGER_LOAD = (
    selectinload(Job.company),
    selectinload(Job.location),
    selectinload(Job.skills),
)

_RECOMMENDATION_TARGET_FIELD: dict[RecommendationType, str] = {
    RecommendationType.JOB: "job_id",
    RecommendationType.COMPANY: "company_id",
    RecommendationType.SKILL: "skill_id",
}


class MatchingRepository(BaseRepository[ResumeJobMatch]):
    """Data access for match preferences, resume<->job matches, and
    recommendations — one repository per module, following `SearchRepository`.
    """

    model = ResumeJobMatch

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    # --- preferences ------------------------------------------------------------

    async def get_preference(self, *, user_id: uuid.UUID) -> MatchPreference | None:
        stmt = select(MatchPreference).where(MatchPreference.user_id == user_id)
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def upsert_preference(self, *, user_id: uuid.UUID, **fields: Any) -> MatchPreference:
        preference = await self.get_preference(user_id=user_id)
        if preference is None:
            preference = MatchPreference(user_id=user_id, **fields)
            self.session.add(preference)
        else:
            for key, value in fields.items():
                setattr(preference, key, value)
        await self.session.flush()
        return preference

    async def active_preference_user_ids(self) -> list[uuid.UUID]:
        stmt = select(MatchPreference.user_id).where(MatchPreference.is_active.is_(True))
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    # --- resume context (cross-module read) --------------------------------------

    async def get_resume_version_owned(
        self, *, resume_version_id: uuid.UUID, user_id: uuid.UUID
    ) -> ResumeVersion | None:
        stmt = (
            select(ResumeVersion)
            .join(Resume, Resume.id == ResumeVersion.resume_id)
            .where(
                ResumeVersion.id == resume_version_id,
                Resume.user_id == user_id,
                ResumeVersion.deleted_at.is_(None),
            )
            .options(selectinload(ResumeVersion.parsed_data), selectinload(ResumeVersion.embedding))
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def get_resume_version(self, *, resume_version_id: uuid.UUID) -> ResumeVersion | None:
        """Fetch a resume version by id alone (no owner check) — used by the
        Celery task path, which only receives the version id and needs to
        resolve the owning user itself (via `.resume.user_id`).
        """
        stmt = (
            select(ResumeVersion)
            .where(ResumeVersion.id == resume_version_id, ResumeVersion.deleted_at.is_(None))
            .options(
                selectinload(ResumeVersion.parsed_data),
                selectinload(ResumeVersion.embedding),
                selectinload(ResumeVersion.resume),
            )
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def get_current_resume_version_for_user(
        self, *, user_id: uuid.UUID
    ) -> ResumeVersion | None:
        """Most-recently-updated resume's current version for a user.

        A user may have several `Resume`s; for background batch matching one
        representative resume is needed, so the most recently touched one
        wins rather than matching against all of them.
        """
        stmt = (
            select(ResumeVersion)
            .join(Resume, Resume.id == ResumeVersion.resume_id)
            .where(
                Resume.user_id == user_id,
                ResumeVersion.is_current.is_(True),
                ResumeVersion.deleted_at.is_(None),
                Resume.deleted_at.is_(None),
            )
            .options(selectinload(ResumeVersion.parsed_data), selectinload(ResumeVersion.embedding))
            .order_by(Resume.updated_at.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    # --- job context (cross-module read) ------------------------------------------

    async def get_job(self, *, job_id: uuid.UUID) -> Job | None:
        stmt = (
            select(Job).where(Job.id == job_id, Job.deleted_at.is_(None)).options(*_JOB_EAGER_LOAD)
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().one_or_none()

    async def get_jobs_by_ids(self, job_ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, Job]:
        if not job_ids:
            return {}
        stmt = (
            select(Job)
            .where(Job.id.in_(job_ids), Job.deleted_at.is_(None))
            .options(*_JOB_EAGER_LOAD)
        )
        result = await self.session.execute(stmt)
        jobs = result.scalars().unique().all()
        return {job.id: job for job in jobs}

    @staticmethod
    def _semantic_candidate_stmt(*, embedding: list[float], limit: int):
        """Pure query builder (no execution) so its shape is unit-testable
        without a DB connection — mirrors `SearchRepository._apply_filters`.
        """
        distance = JobEmbedding.embedding.cosine_distance(embedding)
        similarity = (1 - distance).label("cosine_sim")
        return (
            select(Job, similarity)
            .join(JobEmbedding, JobEmbedding.job_id == Job.id)
            .where(Job.status == JobStatus.ACTIVE, Job.deleted_at.is_(None))
            .options(*_JOB_EAGER_LOAD)
            .order_by(distance.asc())
            .limit(limit)
        )

    async def semantic_candidate_jobs(
        self, *, embedding: list[float], limit: int
    ) -> list[tuple[Job, float]]:
        """Top-`limit` active jobs by pgvector cosine similarity to `embedding`."""
        stmt = self._semantic_candidate_stmt(embedding=embedding, limit=limit)
        result = await self.session.execute(stmt)
        return [(job, float(similarity_value)) for job, similarity_value in result.all()]

    async def job_similarity(self, *, embedding: list[float], job_id: uuid.UUID) -> float | None:
        """Cosine similarity between `embedding` and a single job's embedding."""
        distance = JobEmbedding.embedding.cosine_distance(embedding)
        similarity = (1 - distance).label("cosine_sim")
        stmt = select(similarity).where(JobEmbedding.job_id == job_id)
        result = await self.session.execute(stmt)
        value = result.scalar_one_or_none()
        return float(value) if value is not None else None

    async def get_companies_by_ids(
        self, company_ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, Company]:
        if not company_ids:
            return {}
        stmt = select(Company).where(Company.id.in_(company_ids))
        result = await self.session.execute(stmt)
        return {c.id: c for c in result.scalars().all()}

    async def get_skills_by_ids(self, skill_ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, Skill]:
        if not skill_ids:
            return {}
        stmt = select(Skill).where(Skill.id.in_(skill_ids))
        result = await self.session.execute(stmt)
        return {s.id: s for s in result.scalars().all()}

    async def get_skills_by_normalized_names(self, names: Sequence[str]) -> dict[str, Skill]:
        normalized = {n.strip().lower() for n in names if n.strip()}
        if not normalized:
            return {}
        stmt = select(Skill).where(Skill.normalized_name.in_(normalized))
        result = await self.session.execute(stmt)
        return {s.normalized_name: s for s in result.scalars().all()}

    # --- matches ----------------------------------------------------------------------

    async def upsert_match(
        self,
        *,
        resume_version_id: uuid.UUID,
        job_id: uuid.UUID,
        user_id: uuid.UUID,
        **fields: Any,
    ) -> ResumeJobMatch:
        stmt = select(ResumeJobMatch).where(
            ResumeJobMatch.resume_version_id == resume_version_id, ResumeJobMatch.job_id == job_id
        )
        result = await self.session.execute(stmt)
        match = result.scalars().one_or_none()
        if match is None:
            match = ResumeJobMatch(
                resume_version_id=resume_version_id, job_id=job_id, user_id=user_id, **fields
            )
            self.session.add(match)
        else:
            for key, value in fields.items():
                setattr(match, key, value)
        await self.session.flush()
        return match

    async def get_match(self, *, match_id: uuid.UUID, user_id: uuid.UUID) -> ResumeJobMatch | None:
        stmt = select(ResumeJobMatch).where(
            ResumeJobMatch.id == match_id, ResumeJobMatch.user_id == user_id
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def get_match_by_user_and_job(
        self, *, user_id: uuid.UUID, job_id: uuid.UUID
    ) -> ResumeJobMatch | None:
        """Most-recently-updated match for a `(user_id, job_id)` pair — used
        by `search.job_feed_service.JobFeedService.get_why_score`, which
        only has a job id (not the resume_version_id half of the unique
        key) to look up from.
        """
        stmt = (
            select(ResumeJobMatch)
            .where(ResumeJobMatch.user_id == user_id, ResumeJobMatch.job_id == job_id)
            .order_by(ResumeJobMatch.updated_at.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    @staticmethod
    def _matches_for_resume_base_stmt(*, resume_version_id: uuid.UUID, user_id: uuid.UUID):
        """Pure query builder (no execution) — the ownership predicate every
        resume-match read shares, unit-testable without a DB connection.
        """
        return select(ResumeJobMatch).where(
            ResumeJobMatch.resume_version_id == resume_version_id,
            ResumeJobMatch.user_id == user_id,
        )

    async def list_matches_for_resume(
        self, *, resume_version_id: uuid.UUID, user_id: uuid.UUID, offset: int, limit: int
    ) -> tuple[Sequence[ResumeJobMatch], int]:
        base = self._matches_for_resume_base_stmt(
            resume_version_id=resume_version_id, user_id=user_id
        )
        stmt = (
            base.order_by(ResumeJobMatch.match_score.desc().nullslast()).offset(offset).limit(limit)
        )
        result = await self.session.execute(stmt)
        matches = result.scalars().all()

        count_stmt = select(func.count()).select_from(base.subquery())
        total = (await self.session.execute(count_stmt)).scalar_one()
        return matches, total

    async def list_matches_for_user(
        self, *, user_id: uuid.UUID, limit: int
    ) -> Sequence[ResumeJobMatch]:
        stmt = (
            select(ResumeJobMatch)
            .where(ResumeJobMatch.user_id == user_id, ResumeJobMatch.status == MatchStatus.COMPUTED)
            .order_by(ResumeJobMatch.match_score.desc().nullslast())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def mark_stale_before(self, *, cutoff: datetime) -> int:
        stmt = (
            update(ResumeJobMatch)
            .where(
                ResumeJobMatch.status == MatchStatus.COMPUTED, ResumeJobMatch.updated_at < cutoff
            )
            .values(status=MatchStatus.STALE)
        )
        result = await self.session.execute(stmt)
        await self.session.flush()
        return result.rowcount or 0

    # --- recommendations -------------------------------------------------------------

    async def list_recommendations(
        self,
        *,
        user_id: uuid.UUID,
        recommendation_type: RecommendationType,
        status: RecommendationStatus = RecommendationStatus.ACTIVE,
    ) -> Sequence[Recommendation]:
        stmt = (
            select(Recommendation)
            .where(
                Recommendation.user_id == user_id,
                Recommendation.recommendation_type == recommendation_type,
                Recommendation.status == status,
            )
            .order_by(Recommendation.score.desc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_recommendation(
        self, *, recommendation_id: uuid.UUID, user_id: uuid.UUID
    ) -> Recommendation | None:
        stmt = select(Recommendation).where(
            Recommendation.id == recommendation_id, Recommendation.user_id == user_id
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def dismiss_recommendation(self, recommendation: Recommendation) -> Recommendation:
        recommendation.status = RecommendationStatus.DISMISSED
        await self.session.flush()
        return recommendation

    async def replace_active_recommendations(
        self,
        *,
        user_id: uuid.UUID,
        recommendation_type: RecommendationType,
        candidates: list[dict[str, Any]],
    ) -> int:
        """Delete-then-insert the active set for `(user_id, recommendation_type)`,
        excluding any candidate the user has already dismissed — dismissals
        persist across refreshes unless the candidate drops out entirely.
        """
        target_field = _RECOMMENDATION_TARGET_FIELD[recommendation_type]

        dismissed_stmt = select(Recommendation).where(
            Recommendation.user_id == user_id,
            Recommendation.recommendation_type == recommendation_type,
            Recommendation.status == RecommendationStatus.DISMISSED,
        )
        dismissed_rows = (await self.session.execute(dismissed_stmt)).scalars().all()
        dismissed_ids = {getattr(row, target_field) for row in dismissed_rows}

        filtered = [c for c in candidates if c.get(target_field) not in dismissed_ids]

        await self.session.execute(
            delete(Recommendation).where(
                Recommendation.user_id == user_id,
                Recommendation.recommendation_type == recommendation_type,
                Recommendation.status == RecommendationStatus.ACTIVE,
            )
        )
        for candidate in filtered:
            self.session.add(
                Recommendation(
                    user_id=user_id, recommendation_type=recommendation_type, **candidate
                )
            )
        await self.session.flush()
        return len(filtered)
