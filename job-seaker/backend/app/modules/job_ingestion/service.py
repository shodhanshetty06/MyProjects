"""Job Ingestion module — application/use-case layer.

Router-facing use cases only (company CRUD, job reads, manual crawl
trigger). The actual ingestion/normalization/enrichment pipeline lives in
`tasks.py`, mirroring how `resume_parser`'s upload/CRUD live in `service.py`
while parsing/embedding live in `tasks.py`.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.core.logging import get_logger
from app.modules.job_ingestion.models import (
    ATSPlatform,
    Company,
    CompanyIntelligence,
    IngestionRun,
    Job,
    JobStatus,
)
from app.modules.job_ingestion.repository import (
    CompanyIntelligenceRepository,
    CompanyRepository,
    IngestionRunRepository,
    JobRepository,
)
from app.modules.job_ingestion.schemas import (
    CompanyCreateRequest,
    CompanyIntelligenceUpsertRequest,
    CompanyUpdateRequest,
)

logger = get_logger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _normalize_name(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    if not slug:
        raise ValidationError("Company name must contain at least one alphanumeric character.")
    return slug


class CompanyService:
    """Company registration + Company Intelligence configuration."""

    def __init__(
        self,
        company_repository: CompanyRepository,
        intelligence_repository: CompanyIntelligenceRepository,
        ingestion_run_repository: IngestionRunRepository,
    ) -> None:
        self.company_repository = company_repository
        self.intelligence_repository = intelligence_repository
        self.ingestion_run_repository = ingestion_run_repository

    async def register_company(self, payload: CompanyCreateRequest) -> Company:
        normalized_name = _normalize_name(payload.name)
        if await self.company_repository.get_by_normalized_name(normalized_name=normalized_name):
            raise ConflictError("A company with this name is already registered.")

        company = await self.company_repository.create(
            name=payload.name,
            normalized_name=normalized_name,
            website=payload.website,
            logo_url=payload.logo_url,
            industry=payload.industry,
            description=payload.description,
        )
        await self.intelligence_repository.create(
            company_id=company.id,
            ats_platform=payload.intelligence.ats_platform,
            ats_identifier=payload.intelligence.ats_identifier,
            career_url=payload.intelligence.career_url,
            adapter_config=payload.intelligence.adapter_config,
        )
        logger.info(
            "Company registered for ingestion",
            extra={"extra_fields": {"company_id": str(company.id), "name": company.name}},
        )
        return await self.get_company(company.id)

    async def get_company(self, company_id: uuid.UUID) -> Company:
        company = await self.company_repository.get_with_intelligence(company_id)
        if company is None:
            raise NotFoundError("Company not found.")
        return company

    async def list_companies(self, *, offset: int, limit: int) -> tuple[Sequence[Company], int]:
        companies = await self.company_repository.list_all(offset=offset, limit=limit)
        total = await self.company_repository.count_all()
        return companies, total

    async def update_company(self, company_id: uuid.UUID, payload: CompanyUpdateRequest) -> Company:
        company = await self.get_company(company_id)

        updates = {
            field: value
            for field, value in payload.model_dump(
                exclude={"intelligence"}, exclude_unset=True
            ).items()
        }
        if updates:
            await self.company_repository.update(company, **updates)

        if payload.intelligence is not None:
            await self._upsert_intelligence(company_id, payload.intelligence)

        return await self.get_company(company_id)

    async def _upsert_intelligence(
        self, company_id: uuid.UUID, payload: CompanyIntelligenceUpsertRequest
    ) -> CompanyIntelligence:
        existing = await self.intelligence_repository.get_by_company(company_id=company_id)
        fields = payload.model_dump()
        if existing is None:
            return await self.intelligence_repository.create(company_id=company_id, **fields)
        return await self.intelligence_repository.update(existing, **fields)

    async def trigger_crawl(self, company_id: uuid.UUID) -> None:
        company = await self.get_company(company_id)
        if company.intelligence is None or company.intelligence.ats_platform == ATSPlatform.UNKNOWN:
            raise ValidationError(
                "Company has no ATS platform configured — set `intelligence.ats_platform` first.",
                error_code="ats_platform_not_configured",
            )
        self._enqueue_fetch(company_id)

    def _enqueue_fetch(self, company_id: uuid.UUID) -> None:
        from app.modules.job_ingestion.tasks import fetch_company_jobs_task

        fetch_company_jobs_task.delay(str(company_id))

    async def list_ingestion_runs(
        self, company_id: uuid.UUID, *, offset: int, limit: int
    ) -> tuple[Sequence[IngestionRun], int]:
        await self.get_company(company_id)
        runs = await self.ingestion_run_repository.list_by_company(
            company_id=company_id, offset=offset, limit=limit
        )
        total = await self.ingestion_run_repository.count_by_company(company_id=company_id)
        return runs, total

    async def delete_company(self, company_id: uuid.UUID) -> None:
        company = await self.get_company(company_id)
        await self.company_repository.update(company, deleted_at=datetime.now(UTC))


class JobService:
    """Read-only job listing/detail. No search/ranking — plain filters only."""

    def __init__(self, job_repository: JobRepository) -> None:
        self.job_repository = job_repository

    async def list_jobs(
        self,
        *,
        company_id: uuid.UUID | None,
        status: JobStatus | None,
        offset: int,
        limit: int,
    ) -> tuple[Sequence[Job], int]:
        jobs = await self.job_repository.list_all(
            company_id=company_id, status=status, offset=offset, limit=limit
        )
        total = await self.job_repository.count_all(company_id=company_id, status=status)
        return jobs, total

    async def get_job(self, job_id: uuid.UUID) -> Job:
        job = await self.job_repository.get_with_details(job_id)
        if job is None:
            raise NotFoundError("Job not found.")
        return job
