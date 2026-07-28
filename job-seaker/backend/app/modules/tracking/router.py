"""Tracking module — API routes (v1).

Application Tracking System: CRUD over tracked job applications, status
transitions with automatic history, notes, a merged timeline, and
dashboard/statistics aggregates. Every endpoint is scoped to the caller via
`CurrentUserIdDep` (trusted `X-User-Id` header — see `app/api/deps.py`).
"""

from __future__ import annotations

import math
import uuid

from fastapi import APIRouter, Depends

from app.api.deps import CurrentUserIdDep, DBSessionDep
from app.modules.notification.repository import (
    NotificationPreferenceRepository,
    NotificationRepository,
)
from app.modules.notification.service import NotificationService
from app.modules.tracking.models import Application
from app.modules.tracking.repository import (
    ApplicationNoteRepository,
    ApplicationRepository,
    ApplicationStatusHistoryRepository,
)
from app.modules.tracking.schemas import (
    ApplicationCreateRequest,
    ApplicationDetail,
    ApplicationFilters,
    ApplicationNoteCreateRequest,
    ApplicationNoteRead,
    ApplicationNoteUpdateRequest,
    ApplicationRead,
    ApplicationStatistics,
    ApplicationStatusChangeRequest,
    ApplicationStatusHistoryRead,
    ApplicationUpdateRequest,
    DashboardResponse,
    TimelineEntry,
)
from app.modules.tracking.service import TrackingService
from app.schemas.common import BaseResponse, PaginatedResponse, PaginationMeta, PaginationParams

router = APIRouter(prefix="/tracking", tags=["Tracking"])


def get_tracking_service(session: DBSessionDep) -> TrackingService:
    return TrackingService(
        application_repository=ApplicationRepository(session),
        status_history_repository=ApplicationStatusHistoryRepository(session),
        note_repository=ApplicationNoteRepository(session),
        notification_service=NotificationService(
            repository=NotificationRepository(session),
            preference_repository=NotificationPreferenceRepository(session),
        ),
    )


ServiceDep = Depends(get_tracking_service)


def _application_read(application: Application) -> ApplicationRead:
    return ApplicationRead.model_validate(application)


def _application_detail(application: Application) -> ApplicationDetail:
    return ApplicationDetail(
        **ApplicationRead.model_validate(application).model_dump(),
        notes=[ApplicationNoteRead.model_validate(n) for n in application.notes],
        status_history=[
            ApplicationStatusHistoryRead.model_validate(h) for h in application.status_history
        ],
    )


# --- CRUD ------------------------------------------------------------------------------


@router.post("/applications", response_model=BaseResponse[ApplicationDetail], status_code=201)
async def create_application(
    payload: ApplicationCreateRequest,
    user_id: CurrentUserIdDep,
    service: TrackingService = ServiceDep,
) -> BaseResponse[ApplicationDetail]:
    application = await service.create_application(user_id=user_id, payload=payload)
    return BaseResponse(message="Application created.", data=_application_detail(application))


@router.get("/applications", response_model=PaginatedResponse[ApplicationRead])
async def list_applications(
    user_id: CurrentUserIdDep,
    filters: ApplicationFilters = Depends(),
    pagination: PaginationParams = Depends(),
    service: TrackingService = ServiceDep,
) -> PaginatedResponse[ApplicationRead]:
    applications, total = await service.list_applications(
        user_id=user_id, filters=filters, offset=pagination.offset, limit=pagination.page_size
    )
    return PaginatedResponse(
        data=[_application_read(a) for a in applications],
        pagination=PaginationMeta(
            page=pagination.page,
            page_size=pagination.page_size,
            total_items=total,
            total_pages=math.ceil(total / pagination.page_size) if total else 0,
        ),
    )


@router.get("/applications/{application_id}", response_model=BaseResponse[ApplicationDetail])
async def get_application(
    application_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: TrackingService = ServiceDep,
) -> BaseResponse[ApplicationDetail]:
    application = await service.get_application(
        user_id=user_id, application_id=application_id, with_details=True
    )
    return BaseResponse(data=_application_detail(application))


@router.patch("/applications/{application_id}", response_model=BaseResponse[ApplicationDetail])
async def update_application(
    application_id: uuid.UUID,
    payload: ApplicationUpdateRequest,
    user_id: CurrentUserIdDep,
    service: TrackingService = ServiceDep,
) -> BaseResponse[ApplicationDetail]:
    application = await service.update_application(
        user_id=user_id, application_id=application_id, payload=payload
    )
    return BaseResponse(message="Application updated.", data=_application_detail(application))


@router.delete("/applications/{application_id}", response_model=BaseResponse[None])
async def delete_application(
    application_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: TrackingService = ServiceDep,
) -> BaseResponse[None]:
    await service.delete_application(user_id=user_id, application_id=application_id)
    return BaseResponse(message="Application deleted.")


# --- status transitions / history -----------------------------------------------------------


@router.post(
    "/applications/{application_id}/status", response_model=BaseResponse[ApplicationDetail]
)
async def change_application_status(
    application_id: uuid.UUID,
    payload: ApplicationStatusChangeRequest,
    user_id: CurrentUserIdDep,
    service: TrackingService = ServiceDep,
) -> BaseResponse[ApplicationDetail]:
    application = await service.change_status(
        user_id=user_id,
        application_id=application_id,
        new_status=payload.status,
        note=payload.note,
    )
    return BaseResponse(message="Application status updated.", data=_application_detail(application))


@router.get(
    "/applications/{application_id}/status-history",
    response_model=BaseResponse[list[ApplicationStatusHistoryRead]],
)
async def get_application_status_history(
    application_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: TrackingService = ServiceDep,
) -> BaseResponse[list[ApplicationStatusHistoryRead]]:
    history = await service.get_status_history(user_id=user_id, application_id=application_id)
    return BaseResponse(
        data=[ApplicationStatusHistoryRead.model_validate(h) for h in history]
    )


@router.get(
    "/applications/{application_id}/timeline", response_model=BaseResponse[list[TimelineEntry]]
)
async def get_application_timeline(
    application_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: TrackingService = ServiceDep,
) -> BaseResponse[list[TimelineEntry]]:
    timeline = await service.get_timeline(user_id=user_id, application_id=application_id)
    return BaseResponse(data=timeline)


# --- notes ----------------------------------------------------------------------------------


@router.post(
    "/applications/{application_id}/notes",
    response_model=BaseResponse[ApplicationNoteRead],
    status_code=201,
)
async def add_application_note(
    application_id: uuid.UUID,
    payload: ApplicationNoteCreateRequest,
    user_id: CurrentUserIdDep,
    service: TrackingService = ServiceDep,
) -> BaseResponse[ApplicationNoteRead]:
    note = await service.add_note(
        user_id=user_id, application_id=application_id, content=payload.content
    )
    return BaseResponse(message="Note added.", data=ApplicationNoteRead.model_validate(note))


@router.get(
    "/applications/{application_id}/notes",
    response_model=BaseResponse[list[ApplicationNoteRead]],
)
async def list_application_notes(
    application_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: TrackingService = ServiceDep,
) -> BaseResponse[list[ApplicationNoteRead]]:
    notes = await service.list_notes(user_id=user_id, application_id=application_id)
    return BaseResponse(data=[ApplicationNoteRead.model_validate(n) for n in notes])


@router.patch(
    "/applications/{application_id}/notes/{note_id}",
    response_model=BaseResponse[ApplicationNoteRead],
)
async def update_application_note(
    application_id: uuid.UUID,
    note_id: uuid.UUID,
    payload: ApplicationNoteUpdateRequest,
    user_id: CurrentUserIdDep,
    service: TrackingService = ServiceDep,
) -> BaseResponse[ApplicationNoteRead]:
    note = await service.update_note(
        user_id=user_id, application_id=application_id, note_id=note_id, content=payload.content
    )
    return BaseResponse(message="Note updated.", data=ApplicationNoteRead.model_validate(note))


@router.delete("/applications/{application_id}/notes/{note_id}", response_model=BaseResponse[None])
async def delete_application_note(
    application_id: uuid.UUID,
    note_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: TrackingService = ServiceDep,
) -> BaseResponse[None]:
    await service.delete_note(user_id=user_id, application_id=application_id, note_id=note_id)
    return BaseResponse(message="Note deleted.")


# --- dashboard / statistics -------------------------------------------------------------------


@router.get("/statistics", response_model=BaseResponse[ApplicationStatistics])
async def get_statistics(
    user_id: CurrentUserIdDep, service: TrackingService = ServiceDep
) -> BaseResponse[ApplicationStatistics]:
    statistics = await service.get_statistics(user_id=user_id)
    return BaseResponse(data=statistics)


@router.get("/dashboard", response_model=BaseResponse[DashboardResponse])
async def get_dashboard(
    user_id: CurrentUserIdDep, service: TrackingService = ServiceDep
) -> BaseResponse[DashboardResponse]:
    statistics, recent_applications, needs_follow_up = await service.get_dashboard(user_id=user_id)
    return BaseResponse(
        data=DashboardResponse(
            statistics=statistics,
            recent_applications=[_application_read(a) for a in recent_applications],
            needs_follow_up=[_application_read(a) for a in needs_follow_up],
        )
    )
