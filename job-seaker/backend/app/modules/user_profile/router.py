"""User Profile module — API routes (v1).

Two routers: the original `/users` status scaffold, and the onboarding
wizard's `/onboarding` endpoints (`POST /onboarding/complete`,
`GET /onboarding/status`) that the frontend gates its post-login redirect on.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import CurrentUserIdDep, DBSessionDep
from app.core.constants import APIVersion
from app.modules.matching.repository import MatchingRepository
from app.modules.matching.service import MatchingService
from app.modules.resume_parser.repository import ResumePipelineRunRepository
from app.schemas.common import BaseResponse
from app.modules.user_profile.repository import UserProfileRepository
from app.modules.user_profile.schemas import (
    OnboardingCompleteRequest,
    OnboardingCompleteResponse,
    OnboardingStatusRead,
    UserProfileRead,
)
from app.modules.user_profile.service import UserProfileService

router = APIRouter(prefix="/users", tags=["User Profile"])


@router.get("/status", response_model=BaseResponse)
async def module_status() -> BaseResponse:
    """Reports that this module is wired up."""
    return BaseResponse(
        message="User Profile module is registered.",
        data={"module": "user_profile", "api_version": APIVersion.V1, "implemented": True},
    )


onboarding_router = APIRouter(prefix="/onboarding", tags=["Onboarding"])


def get_user_profile_service(session: DBSessionDep) -> UserProfileService:
    return UserProfileService(
        repository=UserProfileRepository(session),
        matching_service=MatchingService(MatchingRepository(session)),
    )


ServiceDep = Depends(get_user_profile_service)


@onboarding_router.get("/status", response_model=BaseResponse[OnboardingStatusRead])
async def get_onboarding_status(
    user_id: CurrentUserIdDep,
    session: DBSessionDep,
    service: UserProfileService = ServiceDep,
) -> BaseResponse[OnboardingStatusRead]:
    profile = await service.get_profile(user_id=user_id)
    matching_repo = MatchingRepository(session)
    resume_version = await matching_repo.get_current_resume_version_for_user(user_id=user_id)

    pipeline_run_id = None
    pipeline_status = None
    if resume_version is not None:
        run = await ResumePipelineRunRepository(session).get_by_resume_version(
            resume_version_id=resume_version.id
        )
        if run is not None:
            pipeline_run_id = run.id
            pipeline_status = run.status.value

    return BaseResponse(
        data=OnboardingStatusRead(
            onboarding_completed=profile is not None and profile.onboarding_completed_at is not None,
            has_resume=resume_version is not None,
            pipeline_run_id=pipeline_run_id,
            pipeline_status=pipeline_status,
        )
    )


@onboarding_router.post("/complete", response_model=BaseResponse[OnboardingCompleteResponse])
async def complete_onboarding(
    payload: OnboardingCompleteRequest,
    user_id: CurrentUserIdDep,
    session: DBSessionDep,
    service: UserProfileService = ServiceDep,
) -> BaseResponse[OnboardingCompleteResponse]:
    matching_repo = MatchingRepository(session)
    resume_version = await matching_repo.get_current_resume_version_for_user(user_id=user_id)

    profile = await service.complete_onboarding(
        user_id=user_id, data=payload, has_resume=resume_version is not None
    )
    await session.commit()

    pipeline_run_id = None
    if resume_version is not None:
        run = await ResumePipelineRunRepository(session).get_by_resume_version(
            resume_version_id=resume_version.id
        )
        if run is not None:
            pipeline_run_id = run.id

        # Target roles are now known — (re-)trigger the AI Career Coach
        # insights (skill gap, learning recs, career health) that the
        # automatic pipeline chain almost always skipped earlier, since
        # onboarding hadn't completed yet when it ran. Safe to call
        # unconditionally: insight generation and its downstream hand-off
        # are idempotent.
        from app.modules.ai_career_coach.tasks import compute_dashboard_insights_task

        compute_dashboard_insights_task.delay(str(resume_version.id), str(user_id))

    return BaseResponse(
        message="Onboarding completed.",
        data=OnboardingCompleteResponse(
            profile=UserProfileRead.model_validate(profile), pipeline_run_id=pipeline_run_id
        ),
    )
