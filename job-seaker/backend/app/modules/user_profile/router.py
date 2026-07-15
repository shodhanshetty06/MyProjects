"""User Profile module — API routes (v1)."""
from __future__ import annotations

from fastapi import APIRouter

from app.core.constants import APIVersion
from app.schemas.common import BaseResponse

router = APIRouter(prefix="/users", tags=["User Profile"])


@router.get("/status", response_model=BaseResponse)
async def module_status() -> BaseResponse:
    """Reports that this module is wired up. No business logic in Phase 0."""
    return BaseResponse(
        message="User Profile module scaffold is registered.",
        data={"module": "user_profile", "api_version": APIVersion.V1, "implemented": False},
    )
