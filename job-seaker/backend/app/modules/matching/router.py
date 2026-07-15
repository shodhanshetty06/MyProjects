"""Matching module — API routes (v1) + realtime WebSocket."""

from __future__ import annotations

import asyncio
import math
import uuid

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from app.api.deps import CurrentUserIdDep, DBSessionDep
from app.integrations.redis_client import get_redis_client
from app.modules.matching.models import RecommendationType
from app.modules.matching.realtime import channel_name
from app.modules.matching.repository import MatchingRepository
from app.modules.matching.schemas import (
    MatchPreferenceRead,
    MatchPreferenceRequest,
    RecommendationRead,
    ResumeJobMatchRead,
)
from app.modules.matching.service import MatchingService
from app.schemas.common import BaseResponse, PaginatedResponse, PaginationMeta, PaginationParams

router = APIRouter(prefix="/matching", tags=["Matching"])


def get_matching_service(session: DBSessionDep) -> MatchingService:
    return MatchingService(MatchingRepository(session))


ServiceDep = Depends(get_matching_service)


# --- preferences --------------------------------------------------------------------


@router.get("/preferences", response_model=BaseResponse[MatchPreferenceRead])
async def get_preferences(
    user_id: CurrentUserIdDep, service: MatchingService = ServiceDep
) -> BaseResponse[MatchPreferenceRead]:
    preference = await service.get_preferences(user_id=user_id)
    return BaseResponse(data=MatchPreferenceRead.model_validate(preference))


@router.put("/preferences", response_model=BaseResponse[MatchPreferenceRead])
async def upsert_preferences(
    payload: MatchPreferenceRequest,
    user_id: CurrentUserIdDep,
    service: MatchingService = ServiceDep,
) -> BaseResponse[MatchPreferenceRead]:
    preference = await service.upsert_preferences(user_id=user_id, data=payload)
    return BaseResponse(
        message="Match preferences saved.", data=MatchPreferenceRead.model_validate(preference)
    )


# --- match computation -----------------------------------------------------------------


@router.post(
    "/resumes/{resume_version_id}/compute",
    response_model=BaseResponse[None],
    status_code=202,
)
async def compute_resume_matches(
    resume_version_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: MatchingService = ServiceDep,
) -> BaseResponse[None]:
    await service.enqueue_batch_compute(user_id=user_id, resume_version_id=resume_version_id)
    return BaseResponse(message="Match computation queued.")


@router.get(
    "/resumes/{resume_version_id}/matches", response_model=PaginatedResponse[ResumeJobMatchRead]
)
async def list_resume_matches(
    resume_version_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    pagination: PaginationParams = Depends(),
    service: MatchingService = ServiceDep,
) -> PaginatedResponse[ResumeJobMatchRead]:
    matches, total = await service.list_matches(
        user_id=user_id,
        resume_version_id=resume_version_id,
        offset=pagination.offset,
        limit=pagination.page_size,
    )
    return PaginatedResponse(
        data=matches,
        pagination=PaginationMeta(
            page=pagination.page,
            page_size=pagination.page_size,
            total_items=total,
            total_pages=math.ceil(total / pagination.page_size) if total else 0,
        ),
    )


@router.get("/matches/{match_id}", response_model=BaseResponse[ResumeJobMatchRead])
async def get_match(
    match_id: uuid.UUID, user_id: CurrentUserIdDep, service: MatchingService = ServiceDep
) -> BaseResponse[ResumeJobMatchRead]:
    match = await service.get_match(user_id=user_id, match_id=match_id)
    return BaseResponse(data=match)


@router.post(
    "/resumes/{resume_version_id}/jobs/{job_id}/compute",
    response_model=BaseResponse[ResumeJobMatchRead],
)
async def compute_single_match(
    resume_version_id: uuid.UUID,
    job_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: MatchingService = ServiceDep,
) -> BaseResponse[ResumeJobMatchRead]:
    match = await service.compute_match(
        user_id=user_id, resume_version_id=resume_version_id, job_id=job_id
    )
    return BaseResponse(message="Match computed.", data=match)


# --- recommendations -----------------------------------------------------------------------


@router.get("/recommendations/jobs", response_model=BaseResponse[list[RecommendationRead]])
async def list_job_recommendations(
    user_id: CurrentUserIdDep, service: MatchingService = ServiceDep
) -> BaseResponse[list[RecommendationRead]]:
    recommendations = await service.list_recommendations(
        user_id=user_id, recommendation_type=RecommendationType.JOB
    )
    return BaseResponse(data=[RecommendationRead.model_validate(r) for r in recommendations])


@router.get("/recommendations/companies", response_model=BaseResponse[list[RecommendationRead]])
async def list_company_recommendations(
    user_id: CurrentUserIdDep, service: MatchingService = ServiceDep
) -> BaseResponse[list[RecommendationRead]]:
    recommendations = await service.list_recommendations(
        user_id=user_id, recommendation_type=RecommendationType.COMPANY
    )
    return BaseResponse(data=[RecommendationRead.model_validate(r) for r in recommendations])


@router.get("/recommendations/skills", response_model=BaseResponse[list[RecommendationRead]])
async def list_skill_recommendations(
    user_id: CurrentUserIdDep, service: MatchingService = ServiceDep
) -> BaseResponse[list[RecommendationRead]]:
    recommendations = await service.list_recommendations(
        user_id=user_id, recommendation_type=RecommendationType.SKILL
    )
    return BaseResponse(data=[RecommendationRead.model_validate(r) for r in recommendations])


@router.post("/recommendations/{recommendation_id}/dismiss", response_model=BaseResponse[None])
async def dismiss_recommendation(
    recommendation_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: MatchingService = ServiceDep,
) -> BaseResponse[None]:
    await service.dismiss_recommendation(user_id=user_id, recommendation_id=recommendation_id)
    return BaseResponse(message="Recommendation dismissed.")


@router.post("/recommendations/refresh", response_model=BaseResponse[None], status_code=202)
async def refresh_recommendations(
    user_id: CurrentUserIdDep, service: MatchingService = ServiceDep
) -> BaseResponse[None]:
    await service.enqueue_recommendations_refresh(user_id=user_id)
    return BaseResponse(message="Recommendation refresh queued.")


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
async def matching_updates_ws(websocket: WebSocket) -> None:
    """Realtime match/recommendation updates for one user.

    PLACEHOLDER auth: `user_id` is read from the query string (a browser
    WebSocket can't carry the `X-User-Id` header used by the REST endpoints)
    — same trust-the-caller stand-in as `app.api.deps.get_current_user_id`
    until real auth exists. Redis (see `app.modules.matching.realtime`) is
    the transport that lets the Celery worker process notify this API
    process's connections.
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
    await pubsub.subscribe(channel_name(user_id))

    forward_task = asyncio.create_task(_forward_redis_to_ws(websocket, pubsub))
    disconnect_task = asyncio.create_task(_watch_for_disconnect(websocket))
    try:
        await asyncio.wait({forward_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED)
    except WebSocketDisconnect:
        pass
    finally:
        forward_task.cancel()
        disconnect_task.cancel()
        await pubsub.unsubscribe(channel_name(user_id))
        await pubsub.aclose()
