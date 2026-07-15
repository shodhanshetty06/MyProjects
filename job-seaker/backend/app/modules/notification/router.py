"""Notification module — API routes (v1).

In-app notification feed (list/read) and per-user notification preferences.
Every endpoint is scoped to the caller via `CurrentUserIdDep` (trusted
`X-User-Id` header — see `app/api/deps.py`).
"""

from __future__ import annotations

import math
import uuid

from fastapi import APIRouter, Depends, Query

from app.api.deps import CurrentUserIdDep, DBSessionDep
from app.modules.notification.repository import (
    NotificationPreferenceRepository,
    NotificationRepository,
)
from app.modules.notification.schemas import (
    NotificationPreferenceRead,
    NotificationPreferenceUpdateRequest,
    NotificationRead,
)
from app.modules.notification.service import NotificationService
from app.schemas.common import BaseResponse, PaginatedResponse, PaginationMeta, PaginationParams

router = APIRouter(prefix="/notifications", tags=["Notification"])


def get_notification_service(session: DBSessionDep) -> NotificationService:
    return NotificationService(
        repository=NotificationRepository(session),
        preference_repository=NotificationPreferenceRepository(session),
    )


ServiceDep = Depends(get_notification_service)


@router.get("", response_model=PaginatedResponse[NotificationRead])
async def list_notifications(
    user_id: CurrentUserIdDep,
    unread_only: bool = Query(default=False),
    pagination: PaginationParams = Depends(),
    service: NotificationService = ServiceDep,
):
    notifications, total = await service.list_notifications(
        user_id=user_id,
        unread_only=unread_only,
        offset=pagination.offset,
        limit=pagination.page_size,
    )
    return PaginatedResponse(
        data=[NotificationRead.model_validate(n) for n in notifications],
        pagination=PaginationMeta(
            page=pagination.page,
            page_size=pagination.page_size,
            total_items=total,
            total_pages=math.ceil(total / pagination.page_size) if total else 0,
        ),
    )


@router.post("/{notification_id}/read", response_model=BaseResponse[NotificationRead])
async def mark_read(
    notification_id: uuid.UUID, user_id: CurrentUserIdDep, service: NotificationService = ServiceDep
):
    notification = await service.mark_read(user_id=user_id, notification_id=notification_id)
    return BaseResponse(data=NotificationRead.model_validate(notification))


@router.post("/read-all", response_model=BaseResponse[dict])
async def mark_all_read(user_id: CurrentUserIdDep, service: NotificationService = ServiceDep):
    count = await service.mark_all_read(user_id=user_id)
    return BaseResponse(data={"marked_read": count})


@router.get("/preferences", response_model=BaseResponse[list[NotificationPreferenceRead]])
async def get_preferences(user_id: CurrentUserIdDep, service: NotificationService = ServiceDep):
    preferences = await service.get_preferences(user_id=user_id)
    return BaseResponse(data=[NotificationPreferenceRead.model_validate(p) for p in preferences])


@router.put("/preferences", response_model=BaseResponse[list[NotificationPreferenceRead]])
async def update_preferences(
    payload: NotificationPreferenceUpdateRequest,
    user_id: CurrentUserIdDep,
    service: NotificationService = ServiceDep,
):
    preferences = await service.update_preferences(user_id=user_id, updates=payload.preferences)
    return BaseResponse(data=[NotificationPreferenceRead.model_validate(p) for p in preferences])
