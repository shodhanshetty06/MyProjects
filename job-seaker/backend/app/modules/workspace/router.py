"""Workspace module — API routes (v1) + realtime WebSocket.

The single aggregation surface the frontend's `/workspace` page reads from
— replaces manually visiting each standalone module page. No business logic
lives here or in `service.py`; every field is composed from other modules'
existing repositories/services (see `WorkspaceService`).
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from app.api.deps import CurrentUserIdDep, DBSessionDep
from app.integrations.redis_client import get_redis_client
from app.modules.ai_career_coach.repository import CareerInsightRepository
from app.modules.matching import realtime as matching_realtime
from app.modules.matching.repository import MatchingRepository
from app.modules.notification.repository import (
    NotificationPreferenceRepository,
    NotificationRepository,
)
from app.modules.notification.service import NotificationService
from app.modules.resume_parser.repository import (
    ResumePipelineRunRepository,
    ResumeVersionRepository,
)
from app.modules.search import realtime as search_realtime
from app.modules.tracking.repository import (
    ApplicationNoteRepository,
    ApplicationRepository,
    ApplicationStatusHistoryRepository,
)
from app.modules.tracking.service import TrackingService
from app.modules.user_profile.repository import UserProfileRepository
from app.modules.workspace import realtime as workspace_realtime
from app.modules.workspace.schemas import PipelineStatus, WorkspaceSummary
from app.modules.workspace.service import WorkspaceService
from app.schemas.common import BaseResponse

router = APIRouter(prefix="/workspace", tags=["Workspace"])


def get_workspace_service(session: DBSessionDep) -> WorkspaceService:
    notification_service = NotificationService(
        repository=NotificationRepository(session),
        preference_repository=NotificationPreferenceRepository(session),
    )
    tracking_service = TrackingService(
        application_repository=ApplicationRepository(session),
        status_history_repository=ApplicationStatusHistoryRepository(session),
        note_repository=ApplicationNoteRepository(session),
        notification_service=notification_service,
    )
    return WorkspaceService(
        matching_repository=MatchingRepository(session),
        resume_version_repository=ResumeVersionRepository(session),
        pipeline_run_repository=ResumePipelineRunRepository(session),
        user_profile_repository=UserProfileRepository(session),
        tracking_service=tracking_service,
        career_insight_repository=CareerInsightRepository(session),
        notification_repository=NotificationRepository(session),
    )


ServiceDep = Depends(get_workspace_service)


@router.get("", response_model=BaseResponse[WorkspaceSummary])
async def get_workspace(
    user_id: CurrentUserIdDep, service: WorkspaceService = ServiceDep
) -> BaseResponse[WorkspaceSummary]:
    summary = await service.get_summary(user_id=user_id)
    return BaseResponse(data=summary)


@router.get("/progress", response_model=BaseResponse[PipelineStatus | None])
async def get_workspace_progress(
    user_id: CurrentUserIdDep, service: WorkspaceService = ServiceDep
) -> BaseResponse[PipelineStatus | None]:
    status = await service.get_pipeline_progress(user_id=user_id)
    return BaseResponse(data=status)


# --- realtime WebSocket ---------------------------------------------------------------------


async def _forward_redis_to_ws(websocket: WebSocket, pubsub) -> None:  # noqa: ANN001
    while True:
        message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=30.0)
        if message is not None:
            await websocket.send_text(message["data"])


async def _watch_for_disconnect(websocket: WebSocket) -> None:
    while True:
        await websocket.receive_text()


@router.websocket("/ws")
async def workspace_updates_ws(websocket: WebSocket) -> None:
    """Realtime workspace updates for one user — one socket covers pipeline
    progress (`workspace:updates:*`), match/recommendation updates
    (`matching:updates:*`, the existing channel `matching.router`'s own
    WebSocket already serves), and Job Search feed updates
    (`search:updates:*` — `new_matching_jobs`/`feed_stale`/`job_saved`/
    `job_unsaved`, see `app.modules.search.realtime`), so the frontend only
    opens one connection instead of a fourth dedicated socket (plan section
    7). Same PLACEHOLDER `user_id` query-string auth as
    `matching.router.matching_updates_ws` — a browser WebSocket can't carry
    the `Authorization` header the REST endpoints use.
    """
    raw_user_id = websocket.query_params.get("user_id")
    try:
        user_id = uuid.UUID(raw_user_id) if raw_user_id else None
    except ValueError:
        user_id = None
    if user_id is None:
        await websocket.close(code=4001, reason="user_id query parameter must be a valid UUID.")
        return

    await websocket.accept()
    redis = get_redis_client()
    pubsub = redis.pubsub()
    channels = (
        workspace_realtime.channel_name(user_id),
        matching_realtime.channel_name(user_id),
        search_realtime.channel_name(user_id),
    )
    await pubsub.subscribe(*channels)

    forward_task = asyncio.create_task(_forward_redis_to_ws(websocket, pubsub))
    disconnect_task = asyncio.create_task(_watch_for_disconnect(websocket))
    try:
        await asyncio.wait({forward_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED)
    except WebSocketDisconnect:
        pass
    finally:
        forward_task.cancel()
        disconnect_task.cancel()
        await pubsub.unsubscribe(*channels)
        await pubsub.aclose()
