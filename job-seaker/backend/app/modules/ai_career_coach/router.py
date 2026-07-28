"""AI Career Coach module — API routes (v1)."""
from __future__ import annotations

import math
import uuid

from fastapi import APIRouter, Depends, Query

from app.api.deps import AIServiceDep, CurrentUserIdDep, DBSessionDep
from app.modules.ai_career_coach.models import CareerInsightType
from app.modules.ai_career_coach.repository import (
    CareerInsightRepository,
    CoachConversationRepository,
    CoachMessageRepository,
    JobMarketRepository,
    ResumeContextRepository,
)
from app.modules.ai_career_coach.schemas import (
    CareerRoadmapRequest,
    CareerRoadmapResult,
    CareerSuggestionResult,
    ChatMessageRequest,
    ChatResponse,
    CoachConversationDetail,
    CoachConversationRead,
    CoachMessageRead,
    CareerInsightRead,
    CareerInsightSummary,
    CountryDemandResult,
    InterviewPrepRequest,
    InterviewPrepResult,
    JobReadinessRequest,
    JobReadinessScoreResult,
    LearningRecommendationResult,
    LearningRecommendationsRequest,
    MarketTrendResult,
    ResumeReviewResult,
    SalaryPredictionRequest,
    SalaryPredictionResult,
    SkillGapRequest,
    SkillGapResult,
)
from app.modules.ai_career_coach.service import AICareerCoachService
from app.schemas.common import BaseResponse, PaginatedResponse, PaginationMeta, PaginationParams

router = APIRouter(prefix="/career-coach", tags=["AI Career Coach"])


def get_ai_career_coach_service(
    session: DBSessionDep, ai_service: AIServiceDep
) -> AICareerCoachService:
    return AICareerCoachService(
        insight_repository=CareerInsightRepository(session),
        conversation_repository=CoachConversationRepository(session),
        message_repository=CoachMessageRepository(session),
        resume_context_repository=ResumeContextRepository(session),
        job_market_repository=JobMarketRepository(session),
        ai_service=ai_service,
    )


ServiceDep = Depends(get_ai_career_coach_service)


def _conversation_read(conversation) -> CoachConversationRead:
    return CoachConversationRead.model_validate(conversation)


def _conversation_detail(conversation) -> CoachConversationDetail:
    return CoachConversationDetail(
        **CoachConversationRead.model_validate(conversation).model_dump(),
        messages=[CoachMessageRead.model_validate(m) for m in conversation.messages],
    )


def _insight_summary(insight) -> CareerInsightSummary:
    return CareerInsightSummary.model_validate(insight)


def _insight_read(insight) -> CareerInsightRead:
    return CareerInsightRead(
        **CareerInsightSummary.model_validate(insight).model_dump(),
        input_params=insight.input_params,
        result=insight.result,
        model_used=insight.model_used,
    )


# --- resume-grounded analysis ------------------------------------------------------------


@router.post(
    "/resumes/{resume_version_id}/review", response_model=BaseResponse[ResumeReviewResult]
)
async def review_resume(
    resume_version_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: AICareerCoachService = ServiceDep,
) -> BaseResponse[ResumeReviewResult]:
    result = await service.review_resume(user_id=user_id, resume_version_id=resume_version_id)
    return BaseResponse(message="Resume review generated.", data=result)


@router.post(
    "/resumes/{resume_version_id}/career-suggestions",
    response_model=BaseResponse[CareerSuggestionResult],
)
async def suggest_careers(
    resume_version_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: AICareerCoachService = ServiceDep,
) -> BaseResponse[CareerSuggestionResult]:
    result = await service.suggest_careers(user_id=user_id, resume_version_id=resume_version_id)
    return BaseResponse(message="Career suggestions generated.", data=result)


@router.post(
    "/resumes/{resume_version_id}/skill-gap", response_model=BaseResponse[SkillGapResult]
)
async def analyze_skill_gap(
    resume_version_id: uuid.UUID,
    payload: SkillGapRequest,
    user_id: CurrentUserIdDep,
    service: AICareerCoachService = ServiceDep,
) -> BaseResponse[SkillGapResult]:
    result = await service.analyze_skill_gap(
        user_id=user_id, resume_version_id=resume_version_id, target_role=payload.target_role
    )
    return BaseResponse(message="Skill gap analysis generated.", data=result)


@router.post(
    "/resumes/{resume_version_id}/learning-recommendations",
    response_model=BaseResponse[LearningRecommendationResult],
)
async def recommend_learning(
    resume_version_id: uuid.UUID,
    payload: LearningRecommendationsRequest,
    user_id: CurrentUserIdDep,
    service: AICareerCoachService = ServiceDep,
) -> BaseResponse[LearningRecommendationResult]:
    result = await service.recommend_learning(
        user_id=user_id,
        resume_version_id=resume_version_id,
        target_role=payload.target_role,
        skills=payload.skills,
    )
    return BaseResponse(message="Learning recommendations generated.", data=result)


@router.post(
    "/resumes/{resume_version_id}/roadmap", response_model=BaseResponse[CareerRoadmapResult]
)
async def generate_roadmap(
    resume_version_id: uuid.UUID,
    payload: CareerRoadmapRequest,
    user_id: CurrentUserIdDep,
    service: AICareerCoachService = ServiceDep,
) -> BaseResponse[CareerRoadmapResult]:
    result = await service.generate_roadmap(
        user_id=user_id,
        resume_version_id=resume_version_id,
        target_role=payload.target_role,
        timeframe_months=payload.timeframe_months,
    )
    return BaseResponse(message="Career roadmap generated.", data=result)


@router.post(
    "/resumes/{resume_version_id}/interview-prep", response_model=BaseResponse[InterviewPrepResult]
)
async def prepare_interview(
    resume_version_id: uuid.UUID,
    payload: InterviewPrepRequest,
    user_id: CurrentUserIdDep,
    service: AICareerCoachService = ServiceDep,
) -> BaseResponse[InterviewPrepResult]:
    result = await service.prepare_interview(
        user_id=user_id, resume_version_id=resume_version_id, target_role=payload.target_role
    )
    return BaseResponse(message="Interview preparation generated.", data=result)


@router.post(
    "/resumes/{resume_version_id}/job-readiness",
    response_model=BaseResponse[JobReadinessScoreResult],
)
async def compute_job_readiness(
    resume_version_id: uuid.UUID,
    payload: JobReadinessRequest,
    user_id: CurrentUserIdDep,
    service: AICareerCoachService = ServiceDep,
) -> BaseResponse[JobReadinessScoreResult]:
    result = await service.compute_job_readiness(
        user_id=user_id, resume_version_id=resume_version_id, target_role=payload.target_role
    )
    return BaseResponse(message="Job readiness score computed.", data=result)


# --- salary + market data ------------------------------------------------------------------


@router.post("/salary-prediction", response_model=BaseResponse[SalaryPredictionResult])
async def predict_salary(
    payload: SalaryPredictionRequest,
    user_id: CurrentUserIdDep,
    service: AICareerCoachService = ServiceDep,
) -> BaseResponse[SalaryPredictionResult]:
    result = await service.predict_salary(
        user_id=user_id,
        target_role=payload.target_role,
        country=payload.country,
        years_experience=payload.years_experience,
        resume_version_id=payload.resume_version_id,
    )
    return BaseResponse(message="Salary prediction generated.", data=result)


@router.get("/market-trends", response_model=BaseResponse[MarketTrendResult])
async def analyze_market_trends(
    user_id: CurrentUserIdDep,
    title: str | None = Query(default=None, max_length=255),
    window_days: int = Query(default=90, ge=1, le=365),
    service: AICareerCoachService = ServiceDep,
) -> BaseResponse[MarketTrendResult]:
    result = await service.analyze_market_trends(
        user_id=user_id, title=title, window_days=window_days
    )
    return BaseResponse(message="Market trend analysis generated.", data=result)


@router.get("/country-demand", response_model=BaseResponse[CountryDemandResult])
async def analyze_country_demand(
    user_id: CurrentUserIdDep,
    title: str | None = Query(default=None, max_length=255),
    skill: str | None = Query(default=None, max_length=255),
    limit: int = Query(default=15, ge=1, le=50),
    service: AICareerCoachService = ServiceDep,
) -> BaseResponse[CountryDemandResult]:
    result = await service.analyze_country_demand(
        user_id=user_id, title=title, skill=skill, limit=limit
    )
    return BaseResponse(message="Country demand analysis generated.", data=result)


# --- insight history ------------------------------------------------------------------------


@router.get("/insights", response_model=PaginatedResponse[CareerInsightSummary])
async def list_insights(
    user_id: CurrentUserIdDep,
    insight_type: CareerInsightType | None = Query(default=None),
    pagination: PaginationParams = Depends(),
    service: AICareerCoachService = ServiceDep,
) -> PaginatedResponse[CareerInsightSummary]:
    insights, total = await service.list_insights(
        user_id=user_id,
        insight_type=insight_type,
        offset=pagination.offset,
        limit=pagination.page_size,
    )
    return PaginatedResponse(
        data=[_insight_summary(i) for i in insights],
        pagination=PaginationMeta(
            page=pagination.page,
            page_size=pagination.page_size,
            total_items=total,
            total_pages=math.ceil(total / pagination.page_size) if total else 0,
        ),
    )


@router.get("/insights/{insight_id}", response_model=BaseResponse[CareerInsightRead])
async def get_insight(
    insight_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: AICareerCoachService = ServiceDep,
) -> BaseResponse[CareerInsightRead]:
    insight = await service.get_insight(user_id=user_id, insight_id=insight_id)
    return BaseResponse(data=_insight_read(insight))


# --- AI chat assistant + conversation history -------------------------------------------------


@router.post("/chat", response_model=BaseResponse[ChatResponse])
async def send_chat_message(
    payload: ChatMessageRequest,
    user_id: CurrentUserIdDep,
    service: AICareerCoachService = ServiceDep,
) -> BaseResponse[ChatResponse]:
    conversation, message = await service.send_message(
        user_id=user_id, conversation_id=payload.conversation_id, message=payload.message
    )
    return BaseResponse(
        data=ChatResponse(
            conversation=_conversation_read(conversation),
            message=CoachMessageRead.model_validate(message),
        )
    )


@router.get("/conversations", response_model=PaginatedResponse[CoachConversationRead])
async def list_conversations(
    user_id: CurrentUserIdDep,
    pagination: PaginationParams = Depends(),
    service: AICareerCoachService = ServiceDep,
) -> PaginatedResponse[CoachConversationRead]:
    conversations, total = await service.list_conversations(
        user_id=user_id, offset=pagination.offset, limit=pagination.page_size
    )
    return PaginatedResponse(
        data=[_conversation_read(c) for c in conversations],
        pagination=PaginationMeta(
            page=pagination.page,
            page_size=pagination.page_size,
            total_items=total,
            total_pages=math.ceil(total / pagination.page_size) if total else 0,
        ),
    )


@router.get(
    "/conversations/{conversation_id}", response_model=BaseResponse[CoachConversationDetail]
)
async def get_conversation(
    conversation_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: AICareerCoachService = ServiceDep,
) -> BaseResponse[CoachConversationDetail]:
    conversation = await service.get_conversation(user_id=user_id, conversation_id=conversation_id)
    return BaseResponse(data=_conversation_detail(conversation))


@router.delete("/conversations/{conversation_id}", response_model=BaseResponse[None])
async def delete_conversation(
    conversation_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: AICareerCoachService = ServiceDep,
) -> BaseResponse[None]:
    await service.delete_conversation(user_id=user_id, conversation_id=conversation_id)
    return BaseResponse(message="Conversation deleted.")
