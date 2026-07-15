"""AI Career Coach module — Pydantic request/response contracts.

Result models (`ResumeReviewResult`, `CareerRoadmapResult`, ...) double as
Claude structured-output schemas: `model_json_schema()` on each is handed
directly to `AIService.extract_structured`, so the API response shape and
the extraction contract can never drift apart (same convention as
`resume_parser.schemas.ParsedResumeData`).
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.modules.ai_career_coach.models import CareerInsightType, CoachMessageRole

Priority = Literal["high", "medium", "low"]


# --- Resume review ------------------------------------------------------------------


class SectionFeedback(BaseModel):
    section: str = Field(description="Resume section this feedback applies to")
    feedback: str


class ImprovementSuggestion(BaseModel):
    section: str = Field(description="Resume section the suggestion applies to")
    issue: str = Field(description="What is weak or missing")
    suggestion: str = Field(description="Concrete rewrite or addition to make")
    priority: Priority = "medium"


class ResumeReviewResult(BaseModel):
    overall_score: float = Field(ge=0, le=100, description="Overall resume quality, 0-100")
    summary: str = Field(description="2-3 sentence overall impression")
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    section_feedback: list[SectionFeedback] = Field(default_factory=list)
    improvement_suggestions: list[ImprovementSuggestion] = Field(default_factory=list)


# --- Career suggestions --------------------------------------------------------------


class SuggestedRole(BaseModel):
    title: str
    fit_score: float = Field(ge=0, le=100, description="How well this role fits the candidate, 0-100")
    match_reason: str


class CareerSuggestionResult(BaseModel):
    summary: str
    suggested_roles: list[SuggestedRole] = Field(default_factory=list)


# --- Skill gap analysis ----------------------------------------------------------------


class SkillGapItem(BaseModel):
    skill: str
    importance: Priority = "medium"
    reason: str


class SkillGapRequest(BaseModel):
    target_role: str = Field(min_length=1, max_length=255)


class SkillGapResult(BaseModel):
    target_role: str
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[SkillGapItem] = Field(default_factory=list)
    overall_readiness_percent: float = Field(ge=0, le=100)


# --- Learning recommendations -----------------------------------------------------------


class LearningRecommendationsRequest(BaseModel):
    target_role: str = Field(min_length=1, max_length=255)
    skills: list[str] | None = Field(
        default=None, description="Skip gap analysis and generate recommendations for these skills directly"
    )


class LearningRecommendation(BaseModel):
    skill: str
    resource_type: Literal["course", "certification", "book", "project", "practice"]
    title: str
    provider: str | None = None
    description: str
    priority: Priority = "medium"


class LearningRecommendationResult(BaseModel):
    target_role: str
    recommendations: list[LearningRecommendation] = Field(default_factory=list)


# --- Salary prediction -----------------------------------------------------------------


class SalaryPredictionRequest(BaseModel):
    target_role: str = Field(min_length=1, max_length=255)
    country: str | None = Field(default=None, max_length=255)
    years_experience: float | None = Field(default=None, ge=0, le=60)
    resume_version_id: uuid.UUID | None = None


class SalaryPredictionResult(BaseModel):
    target_role: str
    country: str | None
    predicted_min: float
    predicted_max: float
    currency: str
    based_on_market_data: bool
    sample_size: int
    narrative: str


# --- Career roadmap --------------------------------------------------------------------


class CareerRoadmapRequest(BaseModel):
    target_role: str = Field(min_length=1, max_length=255)
    timeframe_months: int = Field(default=12, ge=1, le=60)


class RoadmapMilestone(BaseModel):
    phase: str = Field(description="e.g. 'Months 1-3'")
    goals: list[str] = Field(default_factory=list)
    skills_to_develop: list[str] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list)


class CareerRoadmapResult(BaseModel):
    target_role: str
    timeframe_months: int
    milestones: list[RoadmapMilestone] = Field(default_factory=list)


# --- Interview preparation --------------------------------------------------------------


class InterviewPrepRequest(BaseModel):
    target_role: str = Field(min_length=1, max_length=255)


class InterviewQuestion(BaseModel):
    question: str
    category: Literal["behavioral", "technical", "situational"]
    tip: str


class InterviewPrepResult(BaseModel):
    target_role: str
    questions: list[InterviewQuestion] = Field(default_factory=list)
    general_tips: list[str] = Field(default_factory=list)


# --- Job readiness score ----------------------------------------------------------------


class JobReadinessRequest(BaseModel):
    target_role: str | None = Field(default=None, max_length=255)


class JobReadinessScoreResult(BaseModel):
    overall_score: float = Field(ge=0, le=100)
    resume_quality_score: float = Field(ge=0, le=100)
    skill_match_score: float = Field(ge=0, le=100)
    completeness_score: float = Field(ge=0, le=100)
    narrative: str
    recommendations: list[str] = Field(default_factory=list)


# --- Market trend analysis --------------------------------------------------------------


class SkillDemand(BaseModel):
    skill: str
    job_count: int


class CompanyDemand(BaseModel):
    company: str
    job_count: int


class MarketTrendResult(BaseModel):
    window_days: int
    title: str | None = None
    total_postings: int
    previous_window_postings: int
    growth_percent: float | None = Field(
        default=None, description="null when there is no prior-window data to compare against"
    )
    top_skills: list[SkillDemand] = Field(default_factory=list)
    top_companies: list[CompanyDemand] = Field(default_factory=list)
    narrative: str


# --- Country-based demand ---------------------------------------------------------------


class CountryDemand(BaseModel):
    country: str
    country_code: str | None
    job_count: int
    avg_salary_min: float | None
    avg_salary_max: float | None
    currency: str | None


class CountryDemandResult(BaseModel):
    title: str | None
    skill: str | None
    countries: list[CountryDemand] = Field(default_factory=list)
    narrative: str


# --- Insight history --------------------------------------------------------------------


class CareerInsightSummary(BaseModel):
    id: uuid.UUID
    insight_type: CareerInsightType
    resume_version_id: uuid.UUID | None
    created_at: datetime

    model_config = {"from_attributes": True}


class CareerInsightRead(CareerInsightSummary):
    input_params: dict
    result: dict
    model_used: str


# --- Chat assistant / conversation history ------------------------------------------------


class ChatMessageRequest(BaseModel):
    conversation_id: uuid.UUID | None = None
    message: str = Field(min_length=1, max_length=4000)


class CoachMessageRead(BaseModel):
    id: uuid.UUID
    role: CoachMessageRole
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class CoachConversationRead(BaseModel):
    id: uuid.UUID
    title: str
    last_message_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class CoachConversationDetail(CoachConversationRead):
    messages: list[CoachMessageRead] = Field(default_factory=list)


class ChatResponse(BaseModel):
    conversation: CoachConversationRead
    message: CoachMessageRead


__all__ = [
    "Priority",
    "SectionFeedback",
    "ImprovementSuggestion",
    "ResumeReviewResult",
    "SuggestedRole",
    "CareerSuggestionResult",
    "SkillGapItem",
    "SkillGapRequest",
    "SkillGapResult",
    "LearningRecommendationsRequest",
    "LearningRecommendation",
    "LearningRecommendationResult",
    "SalaryPredictionRequest",
    "SalaryPredictionResult",
    "CareerRoadmapRequest",
    "RoadmapMilestone",
    "CareerRoadmapResult",
    "InterviewPrepRequest",
    "InterviewQuestion",
    "InterviewPrepResult",
    "JobReadinessRequest",
    "JobReadinessScoreResult",
    "SkillDemand",
    "CompanyDemand",
    "MarketTrendResult",
    "CountryDemand",
    "CountryDemandResult",
    "CareerInsightSummary",
    "CareerInsightRead",
    "ChatMessageRequest",
    "CoachMessageRead",
    "CoachConversationRead",
    "CoachConversationDetail",
    "ChatResponse",
]
