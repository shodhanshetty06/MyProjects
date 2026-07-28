"""User Profile module — Pydantic request/response contracts."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.modules.matching.schemas import MatchPreferenceRequest


class OnboardingCompleteRequest(MatchPreferenceRequest):
    """Everything the onboarding wizard collects.

    Job-preference fields (countries, work modes, employment types, salary,
    experience) are inherited from `MatchPreferenceRequest` unchanged and
    passed straight through to `MatchingService.upsert_preferences` — only
    `target_roles`/`work_authorization` are new, onboarding-only fields.
    """

    target_roles: list[str] = Field(default_factory=list, max_length=20)
    work_authorization: str | None = Field(default=None, max_length=100)


class UserProfileRead(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    target_roles: list[str]
    work_authorization: str | None
    onboarding_completed_at: datetime | None

    model_config = {"from_attributes": True}


class OnboardingStatusRead(BaseModel):
    onboarding_completed: bool
    has_resume: bool
    pipeline_run_id: uuid.UUID | None = None
    pipeline_status: str | None = None


class OnboardingCompleteResponse(BaseModel):
    profile: UserProfileRead
    pipeline_run_id: uuid.UUID | None = None


__all__ = [
    "OnboardingCompleteRequest",
    "OnboardingCompleteResponse",
    "OnboardingStatusRead",
    "UserProfileRead",
]
