"""Auto Apply module — application/use-case layer.

Router-facing use cases only (profile upsert, task create/list/retry/cancel).
The actual field-discovery/mapping/submission pipeline lives in `tasks.py`,
mirroring how `job_ingestion`/`ats_resume_generator` split router-facing CRUD
from the Celery-driven pipeline.
"""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from app.core.config import get_settings
from app.core.exceptions import ConflictError, NotFoundError
from app.core.logging import get_logger
from app.integrations.s3_client import S3Storage
from app.modules.auto_apply import resume_selection
from app.modules.auto_apply.models import (
    ApplicantProfile,
    AutoApplyAttempt,
    AutoApplyTask,
    AutoApplyTaskStatus,
)
from app.modules.auto_apply.repository import (
    ApplicantProfileRepository,
    AutoApplyAttemptRepository,
    AutoApplyTaskRepository,
)
from app.modules.auto_apply.schemas import (
    ApplicantProfileUpsertRequest,
    AutoApplyTaskCreateRequest,
    AutoApplyTaskFilters,
)
from app.modules.job_ingestion.models import ATSPlatform, JobStatus

logger = get_logger(__name__)

_RETRYABLE_TASK_STATUSES = (AutoApplyTaskStatus.FAILED, AutoApplyTaskStatus.REQUIRES_REVIEW)
_CANCELLABLE_TASK_STATUSES = (
    AutoApplyTaskStatus.PENDING,
    AutoApplyTaskStatus.QUEUED,
    AutoApplyTaskStatus.RETRYING,
    AutoApplyTaskStatus.REQUIRES_REVIEW,
)


class AutoApplyService:
    """Profile management + task lifecycle (create, list, retry, cancel)."""

    def __init__(
        self,
        profile_repository: ApplicantProfileRepository,
        task_repository: AutoApplyTaskRepository,
        attempt_repository: AutoApplyAttemptRepository,
        s3_storage: S3Storage,
    ) -> None:
        self.profile_repository = profile_repository
        self.task_repository = task_repository
        self.attempt_repository = attempt_repository
        self.s3_storage = s3_storage
        self.settings = get_settings()

    # --- profile ------------------------------------------------------------------

    async def get_profile(self, *, user_id: uuid.UUID) -> ApplicantProfile:
        return await self.profile_repository.get_or_create(user_id=user_id)

    async def upsert_profile(
        self, *, user_id: uuid.UUID, payload: ApplicantProfileUpsertRequest
    ) -> ApplicantProfile:
        profile = await self.profile_repository.get_or_create(user_id=user_id)
        return await self.profile_repository.update(profile, **payload.model_dump())

    # --- task create ----------------------------------------------------------------

    async def create_task(
        self, *, user_id: uuid.UUID, payload: AutoApplyTaskCreateRequest
    ) -> AutoApplyTask:
        job = await self.task_repository.get_job_with_company(job_id=payload.job_id)
        if job is None:
            raise NotFoundError("Job not found.")
        if job.status == JobStatus.CLOSED:
            raise ConflictError("This job posting has closed.")

        existing = await self.task_repository.get_active_for_job(user_id=user_id, job_id=job.id)
        if existing is not None:
            raise ConflictError("An auto-apply task for this job is already in progress.")

        platform = (
            job.company.intelligence.ats_platform
            if job.company and job.company.intelligence
            else ATSPlatform.UNKNOWN
        )

        resume_display_name: str | None = None
        if payload.ats_resume_id is not None or payload.resume_version_id is not None:
            ats_resumes = await self.task_repository.list_ats_resumes_for_user(user_id=user_id)
            resumes = await self.task_repository.list_resumes_for_user(user_id=user_id)
            selection = resume_selection.select_resume(
                ats_resumes=ats_resumes,
                resumes=resumes,
                job=job,
                explicit_ats_resume_id=payload.ats_resume_id,
                explicit_resume_version_id=payload.resume_version_id,
            )
            resume_display_name = selection.display_name

        task = await self.task_repository.create(
            user_id=user_id,
            job_id=job.id,
            company_name=job.company.name if job.company else "Unknown",
            job_title=job.title,
            job_url=job.source_url,
            platform=platform,
            ats_resume_id=payload.ats_resume_id,
            resume_version_id=payload.resume_version_id,
            resume_used_display_name=resume_display_name,
            status=AutoApplyTaskStatus.QUEUED,
            max_attempts=self.settings.AUTO_APPLY_MAX_ATTEMPTS,
        )
        self._enqueue_submit(task.id)
        logger.info(
            "Auto-apply task created",
            extra={"extra_fields": {"task_id": str(task.id), "user_id": str(user_id), "job_id": str(job.id)}},
        )
        return task

    def _enqueue_submit(self, task_id: uuid.UUID) -> None:
        from app.modules.auto_apply.tasks import submit_application_task

        submit_application_task.delay(str(task_id))

    # --- reads ------------------------------------------------------------------------

    async def get_task(self, *, user_id: uuid.UUID, task_id: uuid.UUID) -> AutoApplyTask:
        task = await self.task_repository.get_owned(task_id=task_id, user_id=user_id)
        if task is None:
            raise NotFoundError("Auto-apply task not found.")
        return task

    async def list_tasks(
        self, *, user_id: uuid.UUID, filters: AutoApplyTaskFilters, offset: int, limit: int
    ) -> tuple[Sequence[AutoApplyTask], int]:
        tasks = await self.task_repository.list_filtered(
            user_id=user_id, filters=filters, offset=offset, limit=limit
        )
        total = await self.task_repository.count_filtered(user_id=user_id, filters=filters)
        return tasks, total

    async def list_attempts(self, *, user_id: uuid.UUID, task_id: uuid.UUID) -> Sequence[AutoApplyAttempt]:
        await self.get_task(user_id=user_id, task_id=task_id)
        return await self.attempt_repository.list_for_task(task_id=task_id)

    async def get_attempt(
        self, *, user_id: uuid.UUID, task_id: uuid.UUID, attempt_id: uuid.UUID
    ) -> AutoApplyAttempt:
        await self.get_task(user_id=user_id, task_id=task_id)
        attempt = await self.attempt_repository.get_for_task(attempt_id=attempt_id, task_id=task_id)
        if attempt is None:
            raise NotFoundError("Auto-apply attempt not found.")
        return attempt

    # --- lifecycle -----------------------------------------------------------------

    async def retry_task(self, *, user_id: uuid.UUID, task_id: uuid.UUID) -> AutoApplyTask:
        task = await self.get_task(user_id=user_id, task_id=task_id)
        if task.status not in _RETRYABLE_TASK_STATUSES:
            raise ConflictError(
                f"Task is '{task.status.value}' and cannot be retried right now."
            )
        task = await self.task_repository.update(
            task, status=AutoApplyTaskStatus.QUEUED, error_message=None, missing_fields=[]
        )
        self._enqueue_submit(task.id)
        logger.info("Auto-apply task manually retried", extra={"extra_fields": {"task_id": str(task_id)}})
        return task

    async def cancel_task(self, *, user_id: uuid.UUID, task_id: uuid.UUID) -> AutoApplyTask:
        task = await self.get_task(user_id=user_id, task_id=task_id)
        if task.status not in _CANCELLABLE_TASK_STATUSES:
            raise ConflictError(f"Task is '{task.status.value}' and cannot be cancelled.")
        return await self.task_repository.update(task, status=AutoApplyTaskStatus.CANCELLED)

    # --- screenshots ---------------------------------------------------------------

    async def get_screenshot_download_url(
        self, *, user_id: uuid.UUID, task_id: uuid.UUID, attempt_id: uuid.UUID, screenshot_id: uuid.UUID
    ) -> str:
        await self.get_attempt(user_id=user_id, task_id=task_id, attempt_id=attempt_id)
        screenshot = await self.attempt_repository.get_screenshot(
            screenshot_id=screenshot_id, attempt_id=attempt_id
        )
        if screenshot is None:
            raise NotFoundError("Screenshot not found.")
        return self.s3_storage.generate_presigned_download_url(screenshot.s3_key)


__all__ = ["AutoApplyService"]
