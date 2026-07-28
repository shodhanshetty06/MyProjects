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
    # Multi-factor weighted composite (see `search/ranking.py`) — the
    # default sort for the "Recommended" feed (`JobFeedService._feed_recommended`),
    # replacing plain `match_score DESC`. Not (yet) wired into the generic
    # `/search` SQL-level sort, since ranking needs several post-fetch
    # signals rather than one column.
    RANKED = "ranked"


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


class JobQualityReason(BaseModel):
    label: str
    positive: bool
    weight: float


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
    # Needed so Apply/Generate-ATS-Resume quick actions have data to
    # pre-fill without a second fetch (see plan section 1).
    description: str = ""
    source_url: str = ""
    # Job Quality Score (job_ingestion/quality.py) — 0..100, rule-based, null
    # until the ingestion pipeline's enrichment stage has scored this job.
    quality_score: int | None = None
    quality_reasons: list[JobQualityReason] = Field(default_factory=list)
    seniority: str = "unknown"
    tech_stack: list[str] = Field(default_factory=list)


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


# --- Job Feed composition (Section 3) ------------------------------------------------


class JobMatchContext(BaseModel):
    """Mirrors the subset of `matching.ResumeJobMatch` fields relevant to a
    feed card / detail page — reused, not recomputed (see `job_feed_service`).
    """

    match_score: float | None = None
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    score_breakdown: dict = Field(default_factory=dict)


class JobAISignals(BaseModel):
    """Heuristic v1 signals from `search/heuristics.py` — see that module's
    docstrings for the exact formulas and their documented limitations.
    """

    hiring_probability: float
    application_difficulty: Literal["easy", "moderate", "hard"]
    application_difficulty_score: float
    expected_ats_score: float | None = None


class JobFeedItem(SearchResultItem):
    """The single unified job-card shape the frontend's `JobCard`/
    `JobDetailPage` consume — `SearchResultItem` plus feed-only composition.
    """

    match: JobMatchContext | None = None
    ai_signals: JobAISignals
    is_saved: bool = False
    application_status: str | None = None
    # Populated only on the `recommended` feed (see
    # `JobFeedService._feed_recommended` / `search/ranking.py`) — the
    # per-factor 0-100 inputs plus final weighted score behind this job's
    # rank, so the UI can show "why this is #1." `None` on every other feed
    # mode (saved/applied/trending/all/newest), which aren't ranked.
    ranking_breakdown: dict[str, float | None] | None = None


class JobFeedMode(str, enum.Enum):
    RECOMMENDED = "recommended"
    ALL = "all"
    NEWEST = "newest"
    TRENDING = "trending"
    SAVED = "saved"
    APPLIED = "applied"


class JobFeedFilters(SearchFilters):
    feed_mode: JobFeedMode = JobFeedMode.RECOMMENDED


class SavedJobRead(BaseModel):
    id: uuid.UUID
    job_id: uuid.UUID
    created_at: datetime

    model_config = {"from_attributes": True}


class SkillCheckItem(BaseModel):
    """One row of the interleaved ✔/✖ skill checklist (matched skills first,
    in the job's own skill-list order, then missing skills) — the literal
    "Python ✔ FastAPI ✔ Docker ✔ AWS ✖ Kubernetes ✖" view, built on top of
    the same `matched_skills`/`missing_skills` lists rather than a new
    computation.
    """

    skill: str
    matched: bool


class WhyThisScorePanel(BaseModel):
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    skill_checklist: list[SkillCheckItem] = Field(default_factory=list)
    experience_match: str | None = None
    # No underlying computation exists anywhere in the codebase for
    # education-fit (see `matching.scoring`) — literal, not invented data.
    education_match: str = "Not evaluated yet"
    location_match: str | None = None
    salary_match: str | None = None
    resume_suggestions: list[str] = Field(default_factory=list)
    # LLM-generated one-line "why this fits" summary, cached on the
    # ResumeJobMatch row (`ai_narrative`) so it's computed once per match,
    # not once per request. `None` when no narrative has been generated yet
    # (e.g. AI provider unavailable) — the caller/UI must treat this as
    # "not available," not an empty string worth rendering.
    narrative: str | None = None


class JobSearchHomeSummary(BaseModel):
    greeting: str
    jobs_matching_profile_count: int
    new_jobs_since_yesterday: int
    average_match_pct: float | None = None
    highest_match_pct: float | None = None
    companies_hiring_now: int
    recommended_action: NextBestAction | None = None
    # Backend-computed insight fields (see `JobFeedService.get_home_summary`)
    # — replace `AIInsightsStrip`'s prior client-side aggregation over only
    # the currently-loaded feed page with real numbers over the user's full
    # computed match set, so the copy stays accurate regardless of scroll
    # position or active feed tab.
    strong_match_count: int = 0
    top_missing_skill: str | None = None
    top_missing_skill_count: int = 0
    easy_apply_count: int = 0
    applications_needing_follow_up_count: int = 0


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
    "JobMatchContext",
    "JobAISignals",
    "JobFeedItem",
    "JobFeedMode",
    "JobFeedFilters",
    "SavedJobRead",
    "WhyThisScorePanel",
    "JobSearchHomeSummary",
]

# Deferred import to avoid a search<->workspace import cycle at module load
# time (workspace.schemas doesn't import search, but keeping this at the
# bottom mirrors how `NextBestAction` is only needed for the forward-ref
# above, resolved via `model_rebuild()`).
from app.modules.workspace.schemas import NextBestAction  # noqa: E402

JobSearchHomeSummary.model_rebuild()
