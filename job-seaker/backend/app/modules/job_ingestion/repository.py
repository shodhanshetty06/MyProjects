"""Job Ingestion module — data access layer."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.modules.job_ingestion.models import (
    ATSPlatform,
    Company,
    CompanyIntelligence,
    IngestionRun,
    Job,
    JobStatus,
    Location,
    Skill,
)
from app.repositories.base import BaseRepository


class CompanyRepository(BaseRepository[Company]):
    model = Company

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_by_normalized_name(self, *, normalized_name: str) -> Company | None:
        stmt = select(Company).where(
            Company.normalized_name == normalized_name, Company.deleted_at.is_(None)
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def get_with_intelligence(self, company_id: uuid.UUID) -> Company | None:
        stmt = (
            select(Company)
            .where(Company.id == company_id, Company.deleted_at.is_(None))
            .options(selectinload(Company.intelligence))
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def list_all(self, *, offset: int = 0, limit: int = 20) -> Sequence[Company]:
        stmt = (
            select(Company)
            .where(Company.deleted_at.is_(None))
            .options(selectinload(Company.intelligence))
            .order_by(Company.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().all()

    async def count_all(self) -> int:
        stmt = select(func.count()).select_from(Company).where(Company.deleted_at.is_(None))
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def list_due_for_crawl(self, *, stale_before: datetime) -> Sequence[Company]:
        stmt = (
            select(Company)
            .join(CompanyIntelligence, CompanyIntelligence.company_id == Company.id)
            .where(
                Company.deleted_at.is_(None),
                Company.is_active.is_(True),
                CompanyIntelligence.ats_platform != ATSPlatform.UNKNOWN,
                (CompanyIntelligence.last_crawled_at.is_(None))
                | (CompanyIntelligence.last_crawled_at < stale_before),
            )
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().all()


class CompanyIntelligenceRepository(BaseRepository[CompanyIntelligence]):
    model = CompanyIntelligence

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_by_company(self, *, company_id: uuid.UUID) -> CompanyIntelligence | None:
        stmt = select(CompanyIntelligence).where(CompanyIntelligence.company_id == company_id)
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()


class LocationRepository(BaseRepository[Location]):
    model = Location

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_by_normalized_key(self, *, normalized_key: str) -> Location | None:
        stmt = select(Location).where(Location.normalized_key == normalized_key)
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def get_or_create(
        self,
        *,
        normalized_key: str,
        raw_text: str,
        city: str | None,
        region: str | None,
        country: str | None,
        country_code: str | None,
    ) -> Location:
        existing = await self.get_by_normalized_key(normalized_key=normalized_key)
        if existing is not None:
            return existing
        return await self.create(
            normalized_key=normalized_key,
            raw_text=raw_text,
            city=city,
            region=region,
            country=country,
            country_code=country_code,
        )


class SkillRepository(BaseRepository[Skill]):
    model = Skill

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_by_normalized_name(self, *, normalized_name: str) -> Skill | None:
        stmt = select(Skill).where(Skill.normalized_name == normalized_name)
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def get_or_create(self, *, name: str) -> Skill:
        normalized_name = name.strip().lower()
        existing = await self.get_by_normalized_name(normalized_name=normalized_name)
        if existing is not None:
            return existing
        return await self.create(name=name.strip(), normalized_name=normalized_name)

    async def bulk_get_or_create(self, *, names: list[str]) -> list[Skill]:
        skills: list[Skill] = []
        for name in names:
            if not name or not name.strip():
                continue
            skills.append(await self.get_or_create(name=name))
        return skills


class JobRepository(BaseRepository[Job]):
    model = Job

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_by_source(self, *, source: ATSPlatform, source_job_id: str) -> Job | None:
        stmt = select(Job).where(Job.source == source, Job.source_job_id == source_job_id)
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def get_with_details(self, job_id: uuid.UUID) -> Job | None:
        stmt = (
            select(Job)
            .where(Job.id == job_id, Job.deleted_at.is_(None))
            .options(
                selectinload(Job.location),
                selectinload(Job.skills),
                selectinload(Job.embedding),
                selectinload(Job.company),
            )
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().one_or_none()

    async def list_all(
        self,
        *,
        company_id: uuid.UUID | None = None,
        status: JobStatus | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> Sequence[Job]:
        stmt = select(Job).where(Job.deleted_at.is_(None))
        if company_id is not None:
            stmt = stmt.where(Job.company_id == company_id)
        if status is not None:
            stmt = stmt.where(Job.status == status)
        stmt = (
            stmt.options(
                selectinload(Job.location), selectinload(Job.skills), selectinload(Job.embedding)
            )
            .order_by(Job.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().all()

    async def count_all(
        self, *, company_id: uuid.UUID | None = None, status: JobStatus | None = None
    ) -> int:
        stmt = select(func.count()).select_from(Job).where(Job.deleted_at.is_(None))
        if company_id is not None:
            stmt = stmt.where(Job.company_id == company_id)
        if status is not None:
            stmt = stmt.where(Job.status == status)
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def bulk_close_stale(self, *, company_id: uuid.UUID, seen_before: datetime) -> int:
        """Mark active jobs not observed in the latest crawl as closed.

        Returns the number of rows closed.
        """
        result = await self.session.execute(
            update(Job)
            .where(
                Job.company_id == company_id,
                Job.status == JobStatus.ACTIVE,
                Job.last_seen_at < seen_before,
            )
            .values(status=JobStatus.CLOSED)
        )
        await self.session.flush()
        return result.rowcount or 0


class IngestionRunRepository(BaseRepository[IngestionRun]):
    model = IngestionRun

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def list_by_company(
        self, *, company_id: uuid.UUID, offset: int = 0, limit: int = 20
    ) -> Sequence[IngestionRun]:
        stmt = (
            select(IngestionRun)
            .where(IngestionRun.company_id == company_id)
            .order_by(IngestionRun.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def count_by_company(self, *, company_id: uuid.UUID) -> int:
        stmt = (
            select(func.count())
            .select_from(IngestionRun)
            .where(IngestionRun.company_id == company_id)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one()
