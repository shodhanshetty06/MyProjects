"""Analytics module — Pydantic request/response contracts.

This module is a read-only reporting layer over other modules' tables
(`tracking`, `resume_parser`, `ats_resume_generator`, `job_ingestion`,
`email_tracking`) — the same cross-module read convention `ats_resume_generator`
and `matching` already use. Nothing here is persisted; every schema is a
shape returned by an aggregation query.
"""
from __future__ import annotations

import enum
import uuid
from datetime import date, datetime

from pydantic import BaseModel, Field, model_validator

from app.modules.tracking.models import ApplicationStatus


class TrendInterval(str, enum.Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class AnalyticsFilters(BaseModel):
    """Bound as a query-param dependency: `filters: AnalyticsFilters = Depends()`.

    `date_from`/`date_to` bound every trend/aggregate query; when omitted
    each endpoint falls back to a sensible default window (see
    `AnalyticsService`). `interval` controls trend-chart bucketing.
    """

    date_from: datetime | None = None
    date_to: datetime | None = None
    interval: TrendInterval = TrendInterval.WEEK
    company_name: str | None = Field(default=None, max_length=255)
    status: list[ApplicationStatus] | None = None
    top_n: int = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def _validate(self) -> "AnalyticsFilters":
        if self.date_from is not None and self.date_to is not None and self.date_from > self.date_to:
            raise ValueError("date_from cannot be after date_to")
        return self


class TrendPoint(BaseModel):
    bucket: datetime
    count: int


class TrendSeries(BaseModel):
    interval: TrendInterval
    points: list[TrendPoint]


class AcceptanceRatePoint(BaseModel):
    bucket: datetime
    offers: int
    accepted: int
    acceptance_rate: float


class AcceptanceRateSeries(BaseModel):
    interval: TrendInterval
    points: list[AcceptanceRatePoint]
    overall_offers: int
    overall_accepted: int
    overall_acceptance_rate: float


class ResumePerformance(BaseModel):
    total_resumes: int
    total_versions: int
    parsed_versions: int
    parse_success_rate: float
    average_skills_per_version: float
    versions_trend: list[TrendPoint]


class ScoreDistributionBucket(BaseModel):
    range_label: str
    count: int


class ATSScoreTrendPoint(BaseModel):
    bucket: datetime
    average_ats_score: float | None
    average_keyword_score: float | None
    sample_size: int


class ATSResumePerformance(BaseModel):
    total_ats_resumes: int
    total_versions: int
    completed_versions: int
    completion_rate: float
    average_ats_score: float | None
    average_keyword_score: float | None
    score_trend: list[ATSScoreTrendPoint]
    score_distribution: list[ScoreDistributionBucket]


class SkillCount(BaseModel):
    skill: str
    count: int


class CompanyOutcome(BaseModel):
    company_name: str
    applications: int
    interviews: int
    offers: int


class CountryCount(BaseModel):
    country: str
    count: int


class HiringTrendPoint(BaseModel):
    bucket: datetime
    jobs_posted: int


class HiringCompanyCount(BaseModel):
    company_name: str
    jobs_posted: int


class HiringTrends(BaseModel):
    interval: TrendInterval
    trend: list[HiringTrendPoint]
    top_hiring_companies: list[HiringCompanyCount]


class SkillDemandItem(BaseModel):
    skill: str
    job_count: int
    user_has_skill: bool


class ResponseTimePoint(BaseModel):
    bucket: datetime
    average_days: float | None
    sample_size: int


class ResponseTimeSeries(BaseModel):
    interval: TrendInterval
    overall_average_days: float | None
    points: list[ResponseTimePoint]


class HeatmapCell(BaseModel):
    date: date
    count: int


class ApplicationHeatmap(BaseModel):
    cells: list[HeatmapCell]
    total: int
    current_streak_days: int
    longest_streak_days: int


class AnalyticsSummary(BaseModel):
    total_applications: int
    interview_rate: float
    offer_rate: float
    acceptance_rate: float
    average_response_days: float | None


class AnalyticsDashboardResponse(BaseModel):
    generated_at: datetime
    date_from: datetime | None
    date_to: datetime | None
    interval: TrendInterval

    summary: AnalyticsSummary
    applications_trend: TrendSeries
    interviews_trend: TrendSeries
    offers_trend: TrendSeries
    acceptance_rate: AcceptanceRateSeries
    resume_performance: ResumePerformance
    ats_resume_performance: ATSResumePerformance
    top_skills: list[SkillCount]
    top_companies: list[CompanyOutcome]
    top_countries: list[CountryCount]
    hiring_trends: HiringTrends
    skill_demand: list[SkillDemandItem]
    response_time: ResponseTimeSeries
    heatmap: ApplicationHeatmap


class ExportApplicationRow(BaseModel):
    id: uuid.UUID
    company_name: str
    job_title: str
    status: ApplicationStatus
    source: str
    location: str | None
    applied_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


__all__ = [
    "TrendInterval",
    "AnalyticsFilters",
    "TrendPoint",
    "TrendSeries",
    "AcceptanceRatePoint",
    "AcceptanceRateSeries",
    "ResumePerformance",
    "ScoreDistributionBucket",
    "ATSScoreTrendPoint",
    "ATSResumePerformance",
    "SkillCount",
    "CompanyOutcome",
    "CountryCount",
    "HiringTrendPoint",
    "HiringCompanyCount",
    "HiringTrends",
    "SkillDemandItem",
    "ResponseTimePoint",
    "ResponseTimeSeries",
    "HeatmapCell",
    "ApplicationHeatmap",
    "AnalyticsSummary",
    "AnalyticsDashboardResponse",
    "ExportApplicationRow",
]
