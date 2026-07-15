"""AI Career Coach module — data access layer.

Reads `resume_parser` and `job_ingestion` models directly (cross-module
reads over shared `Base.metadata` — the same convention `matching` and
`search` already use for `job_ingestion`), in addition to owning this
module's own three tables.
"""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.modules.ai_career_coach.models import (
    CareerInsight,
    CareerInsightType,
    CoachConversation,
    CoachMessage,
)
from app.modules.job_ingestion.models import Company, Job, JobStatus, Location, Skill, job_skill
from app.modules.resume_parser.models import Resume, ResumeVersion
from app.repositories.base import BaseRepository


class CareerInsightRepository(BaseRepository[CareerInsight]):
    """Data access for generated, cached analysis results."""

    model = CareerInsight

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_latest(
        self,
        *,
        user_id: uuid.UUID,
        insight_type: CareerInsightType,
        resume_version_id: uuid.UUID | None = None,
    ) -> CareerInsight | None:
        stmt = (
            select(CareerInsight)
            .where(
                CareerInsight.user_id == user_id,
                CareerInsight.insight_type == insight_type,
                CareerInsight.resume_version_id == resume_version_id,
            )
            .order_by(CareerInsight.created_at.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def list_by_user(
        self,
        *,
        user_id: uuid.UUID,
        insight_type: CareerInsightType | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> Sequence[CareerInsight]:
        stmt = select(CareerInsight).where(CareerInsight.user_id == user_id)
        if insight_type is not None:
            stmt = stmt.where(CareerInsight.insight_type == insight_type)
        stmt = stmt.order_by(CareerInsight.created_at.desc()).offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def count_by_user(
        self, *, user_id: uuid.UUID, insight_type: CareerInsightType | None = None
    ) -> int:
        stmt = select(func.count()).select_from(CareerInsight).where(
            CareerInsight.user_id == user_id
        )
        if insight_type is not None:
            stmt = stmt.where(CareerInsight.insight_type == insight_type)
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def get_owned(self, *, user_id: uuid.UUID, insight_id: uuid.UUID) -> CareerInsight | None:
        stmt = select(CareerInsight).where(
            CareerInsight.id == insight_id, CareerInsight.user_id == user_id
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()


class CoachConversationRepository(BaseRepository[CoachConversation]):
    """Data access for chat-assistant threads."""

    model = CoachConversation

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def list_by_user(
        self, *, user_id: uuid.UUID, offset: int = 0, limit: int = 20
    ) -> Sequence[CoachConversation]:
        stmt = (
            select(CoachConversation)
            .where(CoachConversation.user_id == user_id, CoachConversation.deleted_at.is_(None))
            .order_by(func.coalesce(CoachConversation.last_message_at, CoachConversation.created_at).desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def count_by_user(self, *, user_id: uuid.UUID) -> int:
        stmt = select(func.count()).select_from(CoachConversation).where(
            CoachConversation.user_id == user_id, CoachConversation.deleted_at.is_(None)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def get_owned(
        self, *, user_id: uuid.UUID, conversation_id: uuid.UUID, with_messages: bool = False
    ) -> CoachConversation | None:
        stmt = select(CoachConversation).where(
            CoachConversation.id == conversation_id,
            CoachConversation.user_id == user_id,
            CoachConversation.deleted_at.is_(None),
        )
        if with_messages:
            stmt = stmt.options(selectinload(CoachConversation.messages))
        result = await self.session.execute(stmt)
        return result.scalars().unique().one_or_none()


class CoachMessageRepository(BaseRepository[CoachMessage]):
    """Data access for individual chat turns."""

    model = CoachMessage

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def list_recent_for_conversation(
        self, *, conversation_id: uuid.UUID, limit: int
    ) -> Sequence[CoachMessage]:
        stmt = (
            select(CoachMessage)
            .where(CoachMessage.conversation_id == conversation_id)
            .order_by(CoachMessage.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(reversed(result.scalars().all()))


class ResumeContextRepository:
    """Read-only cross-module access into `resume_parser` for coach context.

    A plain class (not `BaseRepository`) since this module never writes to
    `resume_parser`'s tables — only reads them for grounding.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_owned_version(
        self, *, user_id: uuid.UUID, resume_version_id: uuid.UUID
    ) -> ResumeVersion | None:
        stmt = (
            select(ResumeVersion)
            .join(Resume, Resume.id == ResumeVersion.resume_id)
            .where(
                ResumeVersion.id == resume_version_id,
                Resume.user_id == user_id,
                ResumeVersion.deleted_at.is_(None),
                Resume.deleted_at.is_(None),
            )
            .options(selectinload(ResumeVersion.parsed_data))
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def get_current_version_for_user(self, *, user_id: uuid.UUID) -> ResumeVersion | None:
        """Best-effort lookup of the user's most recently updated current resume version."""
        stmt = (
            select(ResumeVersion)
            .join(Resume, Resume.id == ResumeVersion.resume_id)
            .where(
                Resume.user_id == user_id,
                ResumeVersion.is_current.is_(True),
                ResumeVersion.deleted_at.is_(None),
                Resume.deleted_at.is_(None),
            )
            .options(selectinload(ResumeVersion.parsed_data))
            .order_by(Resume.updated_at.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()


class JobMarketRepository:
    """Read-only cross-module access into `job_ingestion` for market-grounded
    features (skill gap, salary prediction, market trends, country demand).

    A plain class (not `BaseRepository`) — this module only reads job
    postings, never writes them.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def required_skills_for_title(
        self, *, title_query: str, limit: int
    ) -> Sequence[tuple[str, int]]:
        """Skills most frequently required across active postings matching `title_query`."""
        job_count = func.count(job_skill.c.job_id).label("job_count")
        stmt = (
            select(Skill.name, job_count)
            .join(job_skill, job_skill.c.skill_id == Skill.id)
            .join(Job, Job.id == job_skill.c.job_id)
            .where(
                Job.status == JobStatus.ACTIVE,
                Job.deleted_at.is_(None),
                Job.title.ilike(f"%{title_query}%"),
            )
            .group_by(Skill.id, Skill.name)
            .order_by(job_count.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return [(row.name, row.job_count) for row in result.all()]

    async def salary_stats_for_role(
        self, *, title_query: str, country: str | None, sample_limit: int
    ) -> dict:
        """Aggregate salary stats over active postings matching `title_query`.

        Only rows with both `salary_min` and `salary_max` set are counted —
        `sample_limit` caps how many matching postings are considered so a
        very broad title query can't scan the whole table.
        """
        subquery = select(Job.id, Job.salary_min, Job.salary_max, Job.salary_currency).where(
            Job.status == JobStatus.ACTIVE,
            Job.deleted_at.is_(None),
            Job.title.ilike(f"%{title_query}%"),
            Job.salary_min.is_not(None),
            Job.salary_max.is_not(None),
        )
        if country:
            subquery = subquery.join(Location, Location.id == Job.location_id).where(
                Location.country.ilike(country)
            )
        subquery = subquery.limit(sample_limit).subquery()

        stmt = select(
            func.count().label("sample_size"),
            func.avg(subquery.c.salary_min).label("avg_min"),
            func.avg(subquery.c.salary_max).label("avg_max"),
            func.mode().within_group(subquery.c.salary_currency).label("currency"),
        )
        result = await self.session.execute(stmt)
        row = result.one()
        return {
            "sample_size": row.sample_size or 0,
            "avg_min": float(row.avg_min) if row.avg_min is not None else None,
            "avg_max": float(row.avg_max) if row.avg_max is not None else None,
            "currency": row.currency,
        }

    async def posting_counts_for_window(
        self, *, window_days: int, title_filter: str | None = None
    ) -> tuple[int, int]:
        """(current_window_count, previous_window_count) of active postings by `posted_at`."""
        now = datetime.now(timezone.utc)
        current_cutoff = now - timedelta(days=window_days)
        previous_cutoff = now - timedelta(days=window_days * 2)

        current_stmt = select(func.count()).select_from(Job).where(
            Job.status == JobStatus.ACTIVE,
            Job.deleted_at.is_(None),
            Job.posted_at >= current_cutoff,
        )
        previous_stmt = select(func.count()).select_from(Job).where(
            Job.status == JobStatus.ACTIVE,
            Job.deleted_at.is_(None),
            Job.posted_at >= previous_cutoff,
            Job.posted_at < current_cutoff,
        )
        if title_filter:
            current_stmt = current_stmt.where(Job.title.ilike(f"%{title_filter}%"))
            previous_stmt = previous_stmt.where(Job.title.ilike(f"%{title_filter}%"))
        current = (await self.session.execute(current_stmt)).scalar_one()
        previous = (await self.session.execute(previous_stmt)).scalar_one()
        return current, previous

    async def top_skills_for_window(
        self, *, window_days: int, limit: int, title_filter: str | None = None
    ) -> Sequence[tuple[str, int]]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        job_count = func.count(job_skill.c.job_id).label("job_count")
        stmt = (
            select(Skill.name, job_count)
            .join(job_skill, job_skill.c.skill_id == Skill.id)
            .join(Job, Job.id == job_skill.c.job_id)
            .where(
                Job.status == JobStatus.ACTIVE, Job.deleted_at.is_(None), Job.posted_at >= cutoff
            )
        )
        if title_filter:
            stmt = stmt.where(Job.title.ilike(f"%{title_filter}%"))
        stmt = stmt.group_by(Skill.id, Skill.name).order_by(job_count.desc()).limit(limit)
        result = await self.session.execute(stmt)
        return [(row.name, row.job_count) for row in result.all()]

    async def top_companies_for_window(
        self, *, window_days: int, limit: int, title_filter: str | None = None
    ) -> Sequence[tuple[str, int]]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        job_count = func.count(Job.id).label("job_count")
        stmt = (
            select(Company.name, job_count)
            .join(Job, Job.company_id == Company.id)
            .where(
                Job.status == JobStatus.ACTIVE, Job.deleted_at.is_(None), Job.posted_at >= cutoff
            )
        )
        if title_filter:
            stmt = stmt.where(Job.title.ilike(f"%{title_filter}%"))
        stmt = stmt.group_by(Company.id, Company.name).order_by(job_count.desc()).limit(limit)
        result = await self.session.execute(stmt)
        return [(row.name, row.job_count) for row in result.all()]

    async def demand_by_country(
        self, *, title_query: str | None, skill_name: str | None, limit: int
    ) -> Sequence[dict]:
        job_count = func.count(func.distinct(Job.id)).label("job_count")
        stmt = (
            select(
                Location.country,
                Location.country_code,
                job_count,
                func.avg(Job.salary_min).label("avg_salary_min"),
                func.avg(Job.salary_max).label("avg_salary_max"),
                func.mode().within_group(Job.salary_currency).label("currency"),
            )
            .join(Location, Location.id == Job.location_id)
            .where(
                Job.status == JobStatus.ACTIVE,
                Job.deleted_at.is_(None),
                Location.country.is_not(None),
            )
        )
        if title_query:
            stmt = stmt.where(Job.title.ilike(f"%{title_query}%"))
        if skill_name:
            stmt = stmt.join(job_skill, job_skill.c.job_id == Job.id).join(
                Skill, Skill.id == job_skill.c.skill_id
            ).where(Skill.normalized_name == skill_name.strip().lower())

        stmt = (
            stmt.group_by(Location.country, Location.country_code)
            .order_by(job_count.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return [
            {
                "country": row.country,
                "country_code": row.country_code,
                "job_count": row.job_count,
                "avg_salary_min": float(row.avg_salary_min) if row.avg_salary_min is not None else None,
                "avg_salary_max": float(row.avg_salary_max) if row.avg_salary_max is not None else None,
                "currency": row.currency,
            }
            for row in result.all()
        ]
