"""User Profile module — application/use-case layer.

Owns onboarding-only fields (`target_roles`, `work_authorization`,
`onboarding_completed_at`). Job-preference fields are delegated straight to
`MatchingService.upsert_preferences` rather than duplicated here — see
`OnboardingCompleteRequest` in `schemas.py`.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.core.exceptions import ConflictError
from app.core.logging import get_logger
from app.modules.matching.schemas import MatchPreferenceRequest
from app.modules.matching.service import MatchingService
from app.modules.user_profile.models import UserProfile
from app.modules.user_profile.repository import UserProfileRepository
from app.modules.user_profile.schemas import OnboardingCompleteRequest

logger = get_logger(__name__)


class UserProfileService:
    """Business logic orchestration for the User Profile module."""

    def __init__(self, repository: UserProfileRepository, matching_service: MatchingService) -> None:
        self.repository = repository
        self.matching_service = matching_service

    async def get_profile(self, *, user_id: uuid.UUID) -> UserProfile | None:
        return await self.repository.get_by_user(user_id=user_id)

    async def complete_onboarding(
        self, *, user_id: uuid.UUID, data: OnboardingCompleteRequest, has_resume: bool
    ) -> UserProfile:
        if not has_resume:
            raise ConflictError(
                "Upload a resume before completing onboarding.", error_code="resume_required"
            )

        preference_fields = data.model_dump(exclude={"target_roles", "work_authorization"})
        await self.matching_service.upsert_preferences(
            user_id=user_id, data=MatchPreferenceRequest(**preference_fields)
        )

        profile = await self.repository.upsert(
            user_id=user_id,
            target_roles=data.target_roles,
            work_authorization=data.work_authorization,
            onboarding_completed_at=datetime.now(timezone.utc),
        )
        logger.info(
            "Onboarding completed", extra={"extra_fields": {"user_id": str(user_id)}}
        )
        return profile


__all__ = ["UserProfileService"]
