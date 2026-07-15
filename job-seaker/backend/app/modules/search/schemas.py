"""Search module — Pydantic request/response contracts."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.modules.job_ingestion.models import EmploymentType, SalaryPeriod, WorkMode


class SearchMode(str, enum.Enum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class SearchSortBy(str, enum.Enum):
    RELEVANCE = "relevance"
    SALARY = "salary"
    MATCH_SCORE = "match_score"
    NEWEST = "newest"
    COMPANY = "company"


class SortDirection(str, enum.Enum):
    ASC = "asc"
    DESC = "desc"


class ExperienceLevel(str, enum.Enum):
    INTERNSHIP = "internship"
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    EXECUTIVE = "executive"


class DatePosted(str, enum.Enum):
    ANY = "any"
    LAST_24H = "last_24h"
    LAST_7D = "last_7d"
    LAST_30D = "last_30d"


class SearchFilters(BaseModel):
    """Bound as a query-param dependency: `filters: SearchFilters = Depends()`.

    Pagination is intentionally not part of this model — reuse
    `app.schemas.common.PaginationParams` separately, exactly like
    resume_parser/job_ingestion do.
    """

    q: str | None = Field(default=None, max_length=512)
    mode: SearchMode = SearchMode.HYBRID

    country: list[str] | None = None
    state: list[str] | None = None
    city: list[str] | None = None
    work_mode: list[WorkMode] | None = None

    salary_min: float | None = Field(default=None, ge=0)
    salary_max: float | None = Field(default=None, ge=0)
    salary_currency: str | None = Field(default=None, min_length=3, max_length=3)
    salary_period: SalaryPeriod | None = None

    experience_level: list[ExperienceLevel] | None = None

    company_id: list[uuid.UUID] | None = None
    company_name: str | None = None

    skills: list[str] | None = None
    skills_match_all: bool = False

    employment_type: list[EmploymentType] | None = None

    date_posted: DatePosted = DatePosted.ANY
    posted_after: datetime | None = None
    posted_before: datetime | None = None

    sort_by: SearchSortBy = SearchSortBy.RELEVANCE
    sort_direction: SortDirection = SortDirection.DESC

    @model_validator(mode="after")
    def _validate(self) -> SearchFilters:
        if self.salary_min is not None and self.salary_max is not None:
            if self.salary_min > self.salary_max:
                raise ValueError("salary_min cannot exceed salary_max")
        if self.sort_by == SearchSortBy.MATCH_SCORE:
            if self.mode == SearchMode.KEYWORD:
                raise ValueError("sort_by=match_score requires mode=semantic or mode=hybrid")
            if not self.q:
                raise ValueError("sort_by=match_score requires a non-empty q")
        # sort_by=relevance with no `q` (plain filter browsing) is allowed —
        # there's nothing to rank by, so the repository falls back to
        # newest-first ordering in that case rather than rejecting the request.
        return self


class SkillTag(BaseModel):
    id: uuid.UUID
    name: str

    model_config = {"from_attributes": True}


class SearchResultCompany(BaseModel):
    id: uuid.UUID
    name: str
    logo_url: str | None
    industry: str | None

    model_config = {"from_attributes": True}


class SearchResultLocation(BaseModel):
    city: str | None
    region: str | None
    country: str | None

    model_config = {"from_attributes": True}


class SearchResultItem(BaseModel):
    id: uuid.UUID
    title: str
    company: SearchResultCompany
    location: SearchResultLocation | None
    work_mode: WorkMode
    employment_type: EmploymentType
    salary_min: float | None
    salary_max: float | None
    salary_currency: str | None
    salary_period: SalaryPeriod
    posted_at: datetime | None
    skills: list[SkillTag]
    experience_level: ExperienceLevel | None = None
    relevance_score: float | None = None
    match_score: float | None = None
    keyword_score: float | None = None


class AutocompleteSuggestion(BaseModel):
    type: Literal["job_title", "company", "skill"]
    label: str
    id: uuid.UUID | None = None


class SearchSuggestion(BaseModel):
    label: str
    type: Literal["skill", "job_title"]


class TrendingSkill(BaseModel):
    skill_id: uuid.UUID
    name: str
    job_count: int


class TrendingJob(BaseModel):
    id: uuid.UUID
    title: str
    company_name: str
    posted_at: datetime | None


__all__ = [
    "SearchMode",
    "SearchSortBy",
    "SortDirection",
    "ExperienceLevel",
    "DatePosted",
    "SearchFilters",
    "SkillTag",
    "SearchResultCompany",
    "SearchResultLocation",
    "SearchResultItem",
    "AutocompleteSuggestion",
    "SearchSuggestion",
    "TrendingSkill",
    "TrendingJob",
]
