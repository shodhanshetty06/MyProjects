"""Job Ingestion module — Pydantic request/response contracts."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.modules.job_ingestion.models import (
    ATSPlatform,
    EmploymentType,
    IngestionRunStatus,
    JobStatus,
    RemotePolicy,
    SalaryPeriod,
    SeniorityLevel,
    WorkMode,
)


class CompanyIntelligenceRead(BaseModel):
    ats_platform: ATSPlatform
    ats_identifier: str | None
    career_url: str | None
    adapter_config: dict
    last_crawled_at: datetime | None
    last_crawl_status: IngestionRunStatus | None
    hiring_frequency_per_month: float | None
    company_rating: float | None
    hiring_trends: dict | None
    company_size: str | None
    funding_stage: str | None
    remote_policy: RemotePolicy
    tech_stack: list[str]

    model_config = {"from_attributes": True}


class CompanyIntelligenceUpsertRequest(BaseModel):
    ats_platform: ATSPlatform = ATSPlatform.UNKNOWN
    ats_identifier: str | None = None
    career_url: str | None = None
    adapter_config: dict = Field(default_factory=dict)
    company_size: str | None = None
    funding_stage: str | None = None
    remote_policy: RemotePolicy = RemotePolicy.UNKNOWN
    tech_stack: list[str] = Field(default_factory=list)


class CompanyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    website: str | None = None
    logo_url: str | None = None
    industry: str | None = None
    description: str | None = None
    intelligence: CompanyIntelligenceUpsertRequest = Field(
        default_factory=CompanyIntelligenceUpsertRequest
    )


class CompanyUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    website: str | None = None
    logo_url: str | None = None
    industry: str | None = None
    description: str | None = None
    is_active: bool | None = None
    intelligence: CompanyIntelligenceUpsertRequest | None = None


class CompanyRead(BaseModel):
    id: uuid.UUID
    name: str
    normalized_name: str
    website: str | None
    logo_url: str | None
    industry: str | None
    description: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime
    intelligence: CompanyIntelligenceRead | None = None
    # Computed from this company's currently-active yearly-salary postings —
    # not a stored column, so it's never stale (see
    # `JobRepository.get_average_yearly_salary`).
    avg_salary_yearly: float | None = None
    avg_salary_currency: str | None = None

    model_config = {"from_attributes": True}


class IngestionRunRead(BaseModel):
    id: uuid.UUID
    company_id: uuid.UUID
    source: ATSPlatform
    status: IngestionRunStatus
    started_at: datetime | None
    finished_at: datetime | None
    jobs_found: int
    jobs_created: int
    jobs_updated: int
    jobs_closed: int
    jobs_failed: int
    retry_count: int
    error_message: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class LocationRead(BaseModel):
    id: uuid.UUID
    raw_text: str
    city: str | None
    region: str | None
    country: str | None
    country_code: str | None

    model_config = {"from_attributes": True}


class SkillRead(BaseModel):
    id: uuid.UUID
    name: str

    model_config = {"from_attributes": True}


class JobQualityReasonRead(BaseModel):
    label: str
    positive: bool
    weight: float


class JobRead(BaseModel):
    id: uuid.UUID
    company_id: uuid.UUID
    location: LocationRead | None
    source: ATSPlatform
    source_url: str
    title: str
    description: str
    employment_type: EmploymentType
    work_mode: WorkMode
    salary_min: float | None
    salary_max: float | None
    salary_currency: str | None
    salary_period: SalaryPeriod
    posted_at: datetime | None
    status: JobStatus
    skills: list[SkillRead]
    has_embedding: bool = False
    created_at: datetime
    updated_at: datetime
    # AI Job Understanding (Feature 3)
    required_skills: list[str]
    preferred_skills: list[str]
    responsibilities: list[str]
    benefits: list[str]
    tech_stack: list[str]
    seniority: SeniorityLevel
    industry: str | None
    department: str | None
    # Job Quality Score (Feature 5)
    quality_score: int | None
    quality_reasons: list[JobQualityReasonRead]

    model_config = {"from_attributes": True}


__all__ = [
    "CompanyIntelligenceRead",
    "CompanyIntelligenceUpsertRequest",
    "CompanyCreateRequest",
    "CompanyUpdateRequest",
    "CompanyRead",
    "IngestionRunRead",
    "LocationRead",
    "SkillRead",
    "JobQualityReasonRead",
    "JobRead",
]
