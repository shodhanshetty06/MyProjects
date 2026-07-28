"""Matching module — Pydantic request/response contracts."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.modules.job_ingestion.models import EmploymentType, WorkMode
from app.modules.matching.models import MatchStatus, RecommendationStatus, RecommendationType


class MatchPreferenceRequest(BaseModel):
    preferred_countries: list[str] = Field(default_factory=list)
    preferred_work_modes: list[WorkMode] = Field(default_factory=list)
    preferred_employment_types: list[EmploymentType] = Field(default_factory=list)
    years_of_experience: int | None = Field(default=None, ge=0, le=60)
    expected_salary_min: float | None = Field(default=None, ge=0)
    expected_salary_max: float | None = Field(default=None, ge=0)
    salary_currency: str | None = Field(default=None, min_length=3, max_length=3)
    is_active: bool = True


class MatchPreferenceRead(MatchPreferenceRequest):
    id: uuid.UUID
    user_id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class JobSummary(BaseModel):
    id: uuid.UUID
    title: str
    company_name: str
    location: str | None = None
    work_mode: WorkMode
    employment_type: EmploymentType
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    source_url: str
    posted_at: datetime | None = None


class ScoreBreakdown(BaseModel):
    semantic_score: float
    skill_match_score: float
    rule_boost_score: float
    boosts: dict[str, float | None]


class SkillCheckItem(BaseModel):
    """One row of the interleaved ✔/✖ skill checklist — see
    `matching.scoring.build_skill_checklist`. Mirrors
    `app.modules.search.schemas.SkillCheckItem`; kept as a separate schema
    (not shared) since `matching` and `search` are independent API surfaces
    that happen to expose the same shape.
    """

    skill: str
    matched: bool


class ResumeJobMatchRead(BaseModel):
    id: uuid.UUID
    resume_version_id: uuid.UUID
    job_id: uuid.UUID
    status: MatchStatus
    match_score: float | None
    matched_skills: list[str]
    missing_skills: list[str]
    skill_checklist: list[SkillCheckItem] = Field(default_factory=list)
    strengths: list[str]
    weaknesses: list[str]
    score_breakdown: ScoreBreakdown
    ai_narrative: str | None = None
    job: JobSummary | None = None
    created_at: datetime
    updated_at: datetime


class RecommendationRead(BaseModel):
    id: uuid.UUID
    recommendation_type: RecommendationType
    job_id: uuid.UUID | None
    company_id: uuid.UUID | None
    skill_id: uuid.UUID | None
    display_name: str
    score: float
    reason: str
    status: RecommendationStatus
    created_at: datetime


class ComputeMatchesResponse(BaseModel):
    resume_version_id: uuid.UUID
    matches_computed: int


class RefreshRecommendationsResponse(BaseModel):
    jobs: int
    companies: int
    skills: int


__all__ = [
    "MatchPreferenceRequest",
    "MatchPreferenceRead",
    "JobSummary",
    "ScoreBreakdown",
    "ResumeJobMatchRead",
    "RecommendationRead",
    "ComputeMatchesResponse",
    "RefreshRecommendationsResponse",
]
