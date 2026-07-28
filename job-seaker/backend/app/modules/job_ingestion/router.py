"""Job Ingestion module — API routes (v1).

Admin/internal surface for registering companies to crawl and reading back
ingested jobs. No auth dependency (matches the module's Phase 0 stub, which
was already unauthenticated) and, per scope, **no search/filter endpoints**
beyond plain equality filters on `company_id`/`status`.
"""

from __future__ import annotations

import math
import uuid

from fastapi import APIRouter, Depends

from app.api.deps import DBSessionDep
from app.modules.job_ingestion.models import Company, Job, JobStatus
from app.modules.job_ingestion.repository import (
    CompanyIntelligenceRepository,
    CompanyRepository,
    IngestionRunRepository,
    JobRepository,
)
from app.modules.job_ingestion.schemas import (
    CompanyCreateRequest,
    CompanyRead,
    CompanyUpdateRequest,
    IngestionRunRead,
    JobQualityReasonRead,
    JobRead,
    LocationRead,
    SkillRead,
)
from app.modules.job_ingestion.service import CompanyService, JobService
from app.schemas.common import BaseResponse, PaginatedResponse, PaginationMeta, PaginationParams

router = APIRouter(prefix="/job-ingestion", tags=["Job Ingestion"])


def get_company_service(session: DBSessionDep) -> CompanyService:
    return CompanyService(
        company_repository=CompanyRepository(session),
        intelligence_repository=CompanyIntelligenceRepository(session),
        ingestion_run_repository=IngestionRunRepository(session),
        job_repository=JobRepository(session),
    )


def get_job_service(session: DBSessionDep) -> JobService:
    return JobService(job_repository=JobRepository(session))


CompanyServiceDep = Depends(get_company_service)
JobServiceDep = Depends(get_job_service)


async def _company_read(company: Company, service: CompanyService) -> CompanyRead:
    avg_salary, avg_salary_currency = await service.get_avg_salary(company.id)
    read = CompanyRead.model_validate(company)
    read.avg_salary_yearly = avg_salary
    read.avg_salary_currency = avg_salary_currency
    return read


def _job_read(job: Job) -> JobRead:
    return JobRead(
        id=job.id,
        company_id=job.company_id,
        location=LocationRead.model_validate(job.location) if job.location else None,
        source=job.source,
        source_url=job.source_url,
        title=job.title,
        description=job.description,
        employment_type=job.employment_type,
        work_mode=job.work_mode,
        salary_min=float(job.salary_min) if job.salary_min is not None else None,
        salary_max=float(job.salary_max) if job.salary_max is not None else None,
        salary_currency=job.salary_currency,
        salary_period=job.salary_period,
        posted_at=job.posted_at,
        status=job.status,
        skills=[SkillRead.model_validate(skill) for skill in job.skills],
        has_embedding=job.embedding is not None,
        created_at=job.created_at,
        updated_at=job.updated_at,
        required_skills=job.required_skills,
        preferred_skills=job.preferred_skills,
        responsibilities=job.responsibilities,
        benefits=job.benefits,
        tech_stack=job.tech_stack,
        seniority=job.seniority,
        industry=job.industry,
        department=job.department,
        quality_score=job.quality_score,
        quality_reasons=[JobQualityReasonRead(**reason) for reason in job.quality_reasons],
    )


@router.post("/companies", response_model=BaseResponse[CompanyRead], status_code=201)
async def register_company(
    payload: CompanyCreateRequest, service: CompanyService = CompanyServiceDep
) -> BaseResponse[CompanyRead]:
    company = await service.register_company(payload)
    return BaseResponse(
        message="Company registered for ingestion.", data=await _company_read(company, service)
    )


@router.get("/companies", response_model=PaginatedResponse[CompanyRead])
async def list_companies(
    pagination: PaginationParams = Depends(), service: CompanyService = CompanyServiceDep
) -> PaginatedResponse[CompanyRead]:
    companies, total = await service.list_companies(
        offset=pagination.offset, limit=pagination.page_size
    )
    return PaginatedResponse(
        data=[await _company_read(c, service) for c in companies],
        pagination=PaginationMeta(
            page=pagination.page,
            page_size=pagination.page_size,
            total_items=total,
            total_pages=math.ceil(total / pagination.page_size) if total else 0,
        ),
    )


@router.get("/companies/{company_id}", response_model=BaseResponse[CompanyRead])
async def get_company(
    company_id: uuid.UUID, service: CompanyService = CompanyServiceDep
) -> BaseResponse[CompanyRead]:
    company = await service.get_company(company_id)
    return BaseResponse(data=await _company_read(company, service))


@router.patch("/companies/{company_id}", response_model=BaseResponse[CompanyRead])
async def update_company(
    company_id: uuid.UUID,
    payload: CompanyUpdateRequest,
    service: CompanyService = CompanyServiceDep,
) -> BaseResponse[CompanyRead]:
    company = await service.update_company(company_id, payload)
    return BaseResponse(message="Company updated.", data=await _company_read(company, service))


@router.delete("/companies/{company_id}", response_model=BaseResponse[None])
async def delete_company(
    company_id: uuid.UUID, service: CompanyService = CompanyServiceDep
) -> BaseResponse[None]:
    await service.delete_company(company_id)
    return BaseResponse(message="Company deleted.")


@router.post("/companies/{company_id}/crawl", response_model=BaseResponse[None], status_code=202)
async def trigger_company_crawl(
    company_id: uuid.UUID, service: CompanyService = CompanyServiceDep
) -> BaseResponse[None]:
    await service.trigger_crawl(company_id)
    return BaseResponse(message="Crawl queued.")


@router.get(
    "/companies/{company_id}/ingestion-runs", response_model=PaginatedResponse[IngestionRunRead]
)
async def list_ingestion_runs(
    company_id: uuid.UUID,
    pagination: PaginationParams = Depends(),
    service: CompanyService = CompanyServiceDep,
) -> PaginatedResponse[IngestionRunRead]:
    runs, total = await service.list_ingestion_runs(
        company_id, offset=pagination.offset, limit=pagination.page_size
    )
    return PaginatedResponse(
        data=[IngestionRunRead.model_validate(run) for run in runs],
        pagination=PaginationMeta(
            page=pagination.page,
            page_size=pagination.page_size,
            total_items=total,
            total_pages=math.ceil(total / pagination.page_size) if total else 0,
        ),
    )


@router.get("/jobs", response_model=PaginatedResponse[JobRead])
async def list_jobs(
    pagination: PaginationParams = Depends(),
    company_id: uuid.UUID | None = None,
    status: JobStatus | None = None,
    service: JobService = JobServiceDep,
) -> PaginatedResponse[JobRead]:
    jobs, total = await service.list_jobs(
        company_id=company_id, status=status, offset=pagination.offset, limit=pagination.page_size
    )
    return PaginatedResponse(
        data=[_job_read(job) for job in jobs],
        pagination=PaginationMeta(
            page=pagination.page,
            page_size=pagination.page_size,
            total_items=total,
            total_pages=math.ceil(total / pagination.page_size) if total else 0,
        ),
    )


@router.get("/jobs/{job_id}", response_model=BaseResponse[JobRead])
async def get_job(job_id: uuid.UUID, service: JobService = JobServiceDep) -> BaseResponse[JobRead]:
    job = await service.get_job(job_id)
    return BaseResponse(data=_job_read(job))
