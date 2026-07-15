"""Auto Apply module — API routes (v1).

Auto Apply Engine: applicant profile management plus the auto-apply task
queue (create/list/retry/cancel), attempt history, screenshots, and logs.
Every endpoint is scoped to the caller via `CurrentUserIdDep` (trusted
`X-User-Id` header — see `app/api/deps.py`).
"""
from __future__ import annotations

import math
import uuid

from fastapi import APIRouter, Depends

from app.api.deps import CurrentUserIdDep, DBSessionDep, S3StorageDep
from app.modules.auto_apply.models import ApplicantProfile, AutoApplyAttempt, AutoApplyTask
from app.modules.auto_apply.repository import (
    ApplicantProfileRepository,
    AutoApplyAttemptRepository,
    AutoApplyTaskRepository,
)
from app.modules.auto_apply.schemas import (
    ApplicantProfileRead,
    ApplicantProfileUpsertRequest,
    AutoApplyAttemptDetail,
    AutoApplyAttemptSummary,
    AutoApplyScreenshotDownloadResponse,
    AutoApplyTaskCreateRequest,
    AutoApplyTaskDetail,
    AutoApplyTaskFilters,
    AutoApplyTaskRead,
)
from app.modules.auto_apply.service import AutoApplyService
from app.schemas.common import BaseResponse, PaginatedResponse, PaginationMeta, PaginationParams

router = APIRouter(prefix="/auto-apply", tags=["Auto Apply"])


def get_auto_apply_service(session: DBSessionDep, s3_storage: S3StorageDep) -> AutoApplyService:
    return AutoApplyService(
        profile_repository=ApplicantProfileRepository(session),
        task_repository=AutoApplyTaskRepository(session),
        attempt_repository=AutoApplyAttemptRepository(session),
        s3_storage=s3_storage,
    )


ServiceDep = Depends(get_auto_apply_service)


def _profile_read(profile: ApplicantProfile) -> ApplicantProfileRead:
    return ApplicantProfileRead.model_validate(profile)


def _task_read(task: AutoApplyTask) -> AutoApplyTaskRead:
    return AutoApplyTaskRead.model_validate(task)


def _task_detail(task: AutoApplyTask) -> AutoApplyTaskDetail:
    return AutoApplyTaskDetail(
        **_task_read(task).model_dump(),
        field_mapping_snapshot=task.field_mapping_snapshot,
        attempts=[AutoApplyAttemptSummary.model_validate(a) for a in task.attempts],
    )


def _attempt_detail(attempt: AutoApplyAttempt) -> AutoApplyAttemptDetail:
    return AutoApplyAttemptDetail.model_validate(attempt)


# --- profile ------------------------------------------------------------------------


@router.get("/profile", response_model=BaseResponse[ApplicantProfileRead])
async def get_profile(
    user_id: CurrentUserIdDep, service: AutoApplyService = ServiceDep
) -> BaseResponse[ApplicantProfileRead]:
    profile = await service.get_profile(user_id=user_id)
    return BaseResponse(data=_profile_read(profile))


@router.put("/profile", response_model=BaseResponse[ApplicantProfileRead])
async def upsert_profile(
    payload: ApplicantProfileUpsertRequest,
    user_id: CurrentUserIdDep,
    service: AutoApplyService = ServiceDep,
) -> BaseResponse[ApplicantProfileRead]:
    profile = await service.upsert_profile(user_id=user_id, payload=payload)
    return BaseResponse(message="Auto Apply profile saved.", data=_profile_read(profile))


# --- tasks --------------------------------------------------------------------------


@router.post("/tasks", response_model=BaseResponse[AutoApplyTaskRead], status_code=201)
async def create_task(
    payload: AutoApplyTaskCreateRequest,
    user_id: CurrentUserIdDep,
    service: AutoApplyService = ServiceDep,
) -> BaseResponse[AutoApplyTaskRead]:
    task = await service.create_task(user_id=user_id, payload=payload)
    return BaseResponse(message="Auto-apply task queued.", data=_task_read(task))


@router.get("/tasks", response_model=PaginatedResponse[AutoApplyTaskRead])
async def list_tasks(
    user_id: CurrentUserIdDep,
    filters: AutoApplyTaskFilters = Depends(),
    pagination: PaginationParams = Depends(),
    service: AutoApplyService = ServiceDep,
) -> PaginatedResponse[AutoApplyTaskRead]:
    tasks, total = await service.list_tasks(
        user_id=user_id, filters=filters, offset=pagination.offset, limit=pagination.page_size
    )
    return PaginatedResponse(
        data=[_task_read(t) for t in tasks],
        pagination=PaginationMeta(
            page=pagination.page,
            page_size=pagination.page_size,
            total_items=total,
            total_pages=math.ceil(total / pagination.page_size) if total else 0,
        ),
    )


@router.get("/tasks/{task_id}", response_model=BaseResponse[AutoApplyTaskDetail])
async def get_task(
    task_id: uuid.UUID, user_id: CurrentUserIdDep, service: AutoApplyService = ServiceDep
) -> BaseResponse[AutoApplyTaskDetail]:
    task = await service.get_task(user_id=user_id, task_id=task_id)
    return BaseResponse(data=_task_detail(task))


@router.post("/tasks/{task_id}/retry", response_model=BaseResponse[AutoApplyTaskRead])
async def retry_task(
    task_id: uuid.UUID, user_id: CurrentUserIdDep, service: AutoApplyService = ServiceDep
) -> BaseResponse[AutoApplyTaskRead]:
    task = await service.retry_task(user_id=user_id, task_id=task_id)
    return BaseResponse(message="Auto-apply task re-queued.", data=_task_read(task))


@router.post("/tasks/{task_id}/cancel", response_model=BaseResponse[AutoApplyTaskRead])
async def cancel_task(
    task_id: uuid.UUID, user_id: CurrentUserIdDep, service: AutoApplyService = ServiceDep
) -> BaseResponse[AutoApplyTaskRead]:
    task = await service.cancel_task(user_id=user_id, task_id=task_id)
    return BaseResponse(message="Auto-apply task cancelled.", data=_task_read(task))


# --- attempts / screenshots / logs ---------------------------------------------------


@router.get("/tasks/{task_id}/attempts", response_model=BaseResponse[list[AutoApplyAttemptSummary]])
async def list_attempts(
    task_id: uuid.UUID, user_id: CurrentUserIdDep, service: AutoApplyService = ServiceDep
) -> BaseResponse[list[AutoApplyAttemptSummary]]:
    attempts = await service.list_attempts(user_id=user_id, task_id=task_id)
    return BaseResponse(data=[AutoApplyAttemptSummary.model_validate(a) for a in attempts])


@router.get(
    "/tasks/{task_id}/attempts/{attempt_id}", response_model=BaseResponse[AutoApplyAttemptDetail]
)
async def get_attempt(
    task_id: uuid.UUID,
    attempt_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: AutoApplyService = ServiceDep,
) -> BaseResponse[AutoApplyAttemptDetail]:
    attempt = await service.get_attempt(user_id=user_id, task_id=task_id, attempt_id=attempt_id)
    return BaseResponse(data=_attempt_detail(attempt))


@router.get(
    "/tasks/{task_id}/attempts/{attempt_id}/screenshots/{screenshot_id}/download",
    response_model=BaseResponse[AutoApplyScreenshotDownloadResponse],
)
async def download_screenshot(
    task_id: uuid.UUID,
    attempt_id: uuid.UUID,
    screenshot_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: AutoApplyService = ServiceDep,
) -> BaseResponse[AutoApplyScreenshotDownloadResponse]:
    url = await service.get_screenshot_download_url(
        user_id=user_id, task_id=task_id, attempt_id=attempt_id, screenshot_id=screenshot_id
    )
    return BaseResponse(data=AutoApplyScreenshotDownloadResponse(download_url=url))
