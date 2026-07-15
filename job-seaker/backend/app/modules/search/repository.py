"""Search module — data access layer.

Queries `job_ingestion`'s `Job`/`Company`/`Location`/`Skill`/`JobEmbedding`
models directly (a cross-module read — those models are already part of the
shared `Base.metadata` via `app/db/base.py`). Search intentionally does not
own any tables of its own beyond the indexes added in
`alembic/versions/0003_search_indexes.py`.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from sqlalchemy import Select, and_, func, not_, nullslast, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.modules.job_ingestion.models import (
    Company,
    CompanyIntelligence,
    Job,
    JobEmbedding,
    JobStatus,
    Location,
    Skill,
    job_skill,
)
from app.modules.search.schemas import ExperienceLevel, SearchFilters, SearchSortBy, SortDirection
from app.modules.search.scoring import EXPERIENCE_LEVEL_TITLE_PATTERNS, date_posted_cutoff
from app.repositories.base import BaseRepository

_JOB_EAGER_LOAD = (
    selectinload(Job.company),
    selectinload(Job.location),
    selectinload(Job.skills),
)


def _sql_order_by(sort_by: SearchSortBy, sort_direction: SortDirection) -> tuple:
    """Order-by clauses for the pure-filter (no `q`) browse path.

    `relevance` has nothing to rank by without a query, so it falls back to
    newest-first here; `match_score` never reaches this path (it requires a
    non-empty `q`, which routes through the query-driven candidate search
    instead — see `SearchFilters._validate`).
    """
    descending = sort_direction == SortDirection.DESC

    if sort_by == SearchSortBy.SALARY:
        col = func.coalesce(Job.salary_max, Job.salary_min)
        return (nullslast(col.desc() if descending else col.asc()),)
    if sort_by == SearchSortBy.COMPANY:
        return (Company.name.desc() if descending else Company.name.asc(),)
    # NEWEST and the RELEVANCE-without-`q` fallback both sort by recency.
    return (nullslast(Job.posted_at.desc() if descending else Job.posted_at.asc()),)


class SearchRepository(BaseRepository[Job]):
    """Query building for filtered, keyword, semantic, and aggregate reads
    over `Job`. Cannot use `BaseRepository.list()`'s equality-only filters —
    every method here builds its own `Select` via `_apply_filters`.
    """

    model = Job

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    # --- shared filter builder -------------------------------------------------

    def _apply_filters(
        self, stmt: Select, filters: SearchFilters, *, join_company: bool = False
    ) -> Select:
        stmt = stmt.where(Job.status == JobStatus.ACTIVE, Job.deleted_at.is_(None))

        if filters.work_mode:
            stmt = stmt.where(Job.work_mode.in_(filters.work_mode))
        if filters.employment_type:
            stmt = stmt.where(Job.employment_type.in_(filters.employment_type))
        if filters.company_id:
            stmt = stmt.where(Job.company_id.in_(filters.company_id))
        if filters.company_name or join_company:
            stmt = stmt.join(Company, Company.id == Job.company_id)
            if filters.company_name:
                stmt = stmt.where(Company.name.ilike(f"%{filters.company_name}%"))
        if filters.country or filters.state or filters.city:
            stmt = stmt.join(Location, Location.id == Job.location_id)
            if filters.country:
                stmt = stmt.where(Location.country.in_(filters.country))
            if filters.state:
                stmt = stmt.where(Location.region.in_(filters.state))
            if filters.city:
                stmt = stmt.where(Location.city.in_(filters.city))
        if filters.salary_currency:
            stmt = stmt.where(Job.salary_currency == filters.salary_currency)
        if filters.salary_period:
            stmt = stmt.where(Job.salary_period == filters.salary_period)
        if filters.salary_min is not None or filters.salary_max is not None:
            # Range-overlap semantics; a job with no disclosed salary can't
            # be matched against a salary filter, so it's excluded here.
            stmt = stmt.where(Job.salary_min.is_not(None), Job.salary_max.is_not(None))
            if filters.salary_min is not None:
                stmt = stmt.where(Job.salary_max >= filters.salary_min)
            if filters.salary_max is not None:
                stmt = stmt.where(Job.salary_min <= filters.salary_max)
        if filters.skills:
            stmt = self._apply_skills_filter(stmt, filters.skills, filters.skills_match_all)
        if filters.experience_level:
            clause = self._experience_level_clause(filters.experience_level)
            if clause is not None:
                stmt = stmt.where(clause)
        cutoff = date_posted_cutoff(filters.date_posted)
        if filters.posted_after is not None:
            cutoff = filters.posted_after if cutoff is None else max(cutoff, filters.posted_after)
        if cutoff is not None:
            stmt = stmt.where(Job.posted_at >= cutoff)
        if filters.posted_before is not None:
            stmt = stmt.where(Job.posted_at <= filters.posted_before)
        return stmt

    def _apply_skills_filter(self, stmt: Select, skills: list[str], match_all: bool) -> Select:
        normalized = [name.strip().lower() for name in skills if name.strip()]
        if not normalized:
            return stmt
        if match_all:
            for name in normalized:
                exists_clause = (
                    select(job_skill.c.job_id)
                    .join(Skill, Skill.id == job_skill.c.skill_id)
                    .where(job_skill.c.job_id == Job.id, Skill.normalized_name == name)
                    .exists()
                )
                stmt = stmt.where(exists_clause)
            return stmt
        exists_clause = (
            select(job_skill.c.job_id)
            .join(Skill, Skill.id == job_skill.c.skill_id)
            .where(job_skill.c.job_id == Job.id, Skill.normalized_name.in_(normalized))
            .exists()
        )
        return stmt.where(exists_clause)

    def _experience_level_clause(self, levels: Sequence[ExperienceLevel]):
        specific_clauses = {
            level: or_(*(Job.title.ilike(f"%{p}%") for p in patterns))
            for level, patterns in EXPERIENCE_LEVEL_TITLE_PATTERNS.items()
        }
        per_level = []
        for level in levels:
            if level == ExperienceLevel.MID:
                per_level.append(and_(*(not_(c) for c in specific_clauses.values())))
            elif level in specific_clauses:
                per_level.append(specific_clauses[level])
        if not per_level:
            return None
        return or_(*per_level)

    # --- pure filter / browse path (no `q`) -------------------------------------

    async def list_filtered(
        self,
        *,
        filters: SearchFilters,
        sort_by: SearchSortBy,
        sort_direction: SortDirection,
        offset: int,
        limit: int,
    ) -> Sequence[Job]:
        needs_company_join = sort_by == SearchSortBy.COMPANY
        stmt = self._apply_filters(select(Job), filters, join_company=needs_company_join)
        stmt = (
            stmt.options(*_JOB_EAGER_LOAD)
            .order_by(*_sql_order_by(sort_by, sort_direction))
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().all()

    async def count_filtered(self, *, filters: SearchFilters) -> int:
        base = select(func.count(func.distinct(Job.id))).select_from(Job)
        stmt = self._apply_filters(base, filters)
        result = await self.session.execute(stmt)
        return result.scalar_one()

    # --- query-driven candidate search -------------------------------------------

    async def search_keyword_candidates(
        self, *, filters: SearchFilters, query_text: str, limit: int
    ) -> list[tuple[uuid.UUID, float]]:
        tsquery = func.plainto_tsquery("english", query_text)
        rank = func.ts_rank_cd(Job.search_vector, tsquery).label("raw_rank")
        stmt = self._apply_filters(select(Job.id, rank), filters)
        stmt = stmt.where(Job.search_vector.op("@@")(tsquery) | Job.title.op("%")(query_text))
        stmt = stmt.order_by(rank.desc()).limit(limit)
        rows = (await self.session.execute(stmt)).all()
        return [(row.id, float(row.raw_rank)) for row in rows]

    async def search_semantic_candidates(
        self, *, filters: SearchFilters, query_embedding: list[float], limit: int
    ) -> list[tuple[uuid.UUID, float]]:
        distance = JobEmbedding.embedding.cosine_distance(query_embedding)
        similarity = (1 - distance).label("cosine_sim")
        stmt = select(Job.id, similarity).join(JobEmbedding, JobEmbedding.job_id == Job.id)
        stmt = self._apply_filters(stmt, filters)
        stmt = stmt.order_by(distance.asc()).limit(limit)
        rows = (await self.session.execute(stmt)).all()
        return [(row.id, float(row.cosine_sim)) for row in rows]

    async def fetch_jobs_by_ids(self, job_ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, Job]:
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

    # --- autocomplete / suggestions ----------------------------------------------

    async def autocomplete_titles(self, *, prefix: str, limit: int) -> list[str]:
        stmt = (
            select(Job.title)
            .distinct()
            .where(Job.status == JobStatus.ACTIVE, Job.title.ilike(f"{prefix}%"))
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def autocomplete_companies(
        self, *, prefix: str, limit: int
    ) -> list[tuple[uuid.UUID, str]]:
        stmt = (
            select(Company.id, Company.name)
            .where(Company.is_active.is_(True), Company.name.ilike(f"{prefix}%"))
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return [(row.id, row.name) for row in result.all()]

    async def autocomplete_skills(self, *, prefix: str, limit: int) -> list[tuple[uuid.UUID, str]]:
        stmt = select(Skill.id, Skill.name).where(Skill.name.ilike(f"{prefix}%")).limit(limit)
        result = await self.session.execute(stmt)
        return [(row.id, row.name) for row in result.all()]

    # --- trending -----------------------------------------------------------------

    async def trending_skills(
        self, *, window_days: int, limit: int
    ) -> list[tuple[uuid.UUID, str, int]]:
        cutoff = datetime.now(UTC) - timedelta(days=window_days)
        job_count = func.count(job_skill.c.job_id).label("job_count")
        stmt = (
            select(Skill.id, Skill.name, job_count)
            .join(job_skill, job_skill.c.skill_id == Skill.id)
            .join(Job, Job.id == job_skill.c.job_id)
            .where(
                Job.status == JobStatus.ACTIVE, Job.deleted_at.is_(None), Job.posted_at >= cutoff
            )
            .group_by(Skill.id, Skill.name)
            .order_by(job_count.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return [(row.id, row.name, row.job_count) for row in result.all()]

    async def trending_jobs(self, *, window_days: int, limit: int) -> Sequence[Job]:
        cutoff = datetime.now(UTC) - timedelta(days=window_days)
        hiring_frequency = CompanyIntelligence.hiring_frequency_per_month
        stmt = (
            select(Job)
            .join(Company, Company.id == Job.company_id)
            .outerjoin(CompanyIntelligence, CompanyIntelligence.company_id == Company.id)
            .where(
                Job.status == JobStatus.ACTIVE, Job.deleted_at.is_(None), Job.posted_at >= cutoff
            )
            .options(selectinload(Job.company), selectinload(Job.location))
            .order_by(Job.posted_at.desc(), hiring_frequency.desc().nullslast())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().all()
