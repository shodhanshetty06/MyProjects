"""Auto Apply module — Celery tasks.

One task, routed to the "auto_apply" queue (see `app/workers/celery_app.py`
`task_routes` and the `worker-auto-apply` service in `docker-compose.yml`):

    submit_application_task -> "auto_apply" queue

Sync-Celery-wrapping-`asyncio.run` convention, same as every other module's
`tasks.py`: a thin wrapper so the existing async DB session
(`AsyncSessionLocal`) can be reused without a parallel sync SQLAlchemy stack.

Retry: only `TransientApplyError`/`PlatformApplyError` (network blips,
timeouts, one-off rendering glitches) are handed to Celery's own
`autoretry_for` + `retry_backoff`, the same convention
`job_ingestion.fetch_company_jobs_task` uses for `httpx.HTTPError`. Every
other failure mode (missing/invalid fields, a CAPTCHA, a duplicate
application) needs a human to act, so it's surfaced via `REQUIRES_REVIEW`/
`FAILED` instead of blindly retried. `max_retries` is set to
`AUTO_APPLY_MAX_ATTEMPTS` so Celery's own retry cap and the task's
`max_attempts` bookkeeping column never disagree.
"""
from __future__ import annotations

import uuid
from asyncio import run as run_async
from datetime import UTC, datetime

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal
from app.dependency_injection.container import get_container
from app.integrations.http_client import build_http_client
from app.interfaces.storage_provider import StorageProvider
from app.modules.auto_apply import field_mapping, resume_selection, validation
from app.modules.auto_apply.adapters import get_apply_adapter
from app.modules.auto_apply.adapters.base import ApplicationField, ApplyResult, ResumeAttachment
from app.modules.auto_apply.errors import (
    PlatformApplyError,
    TransientApplyError,
    classify_exception,
    is_retryable,
)
from app.modules.auto_apply.field_mapping import FieldMappingResult
from app.modules.auto_apply.models import (
    ApplicantProfile,
    AutoApplyAttemptStatus,
    AutoApplyTaskStatus,
    FailureCategory,
)
from app.modules.auto_apply.repository import (
    ApplicantProfileRepository,
    AutoApplyAttemptRepository,
    AutoApplyTaskRepository,
)
from app.modules.auto_apply.validation import ValidationIssue
from app.modules.job_ingestion.models import ATSPlatform
from app.modules.tracking.models import Application, ApplicationSource, ApplicationStatus
from app.modules.tracking.repository import ApplicationRepository, ApplicationStatusHistoryRepository
from app.workers.base_task import BaseTask
from app.workers.celery_app import celery_app

logger = get_logger(__name__)

_settings = get_settings()
_HTTP_PLATFORMS = (ATSPlatform.GREENHOUSE, ATSPlatform.LEVER)


@celery_app.task(
    base=BaseTask,
    bind=True,
    name="auto_apply.submit_application",
    autoretry_for=(TransientApplyError, PlatformApplyError),
    max_retries=_settings.AUTO_APPLY_MAX_ATTEMPTS,
)
def submit_application_task(self, task_id: str) -> None:  # noqa: ANN001 -- Celery `self`
    run_async(_submit_application(uuid.UUID(task_id), retry_count=self.request.retries))


def _map_and_validate(
    fields: list[ApplicationField], profile: ApplicantProfile
) -> tuple[FieldMappingResult, list[ValidationIssue]]:
    field_values = field_mapping.profile_to_field_values(profile)
    mapping_result = field_mapping.map_fields(
        fields, field_values=field_values, custom_answers=profile.custom_answers
    )
    issues = validation.validate_field_values(fields, mapping_result.mapped_values)
    return mapping_result, issues


async def _persist_apply_result(
    *,
    storage_provider: StorageProvider,
    attempt_repo: AutoApplyAttemptRepository,
    attempt_id: uuid.UUID,
    user_id: uuid.UUID,
    task_id: uuid.UUID,
    result: ApplyResult,
) -> None:
    for index, shot in enumerate(result.screenshots):
        key = f"auto-apply/{user_id}/{task_id}/{attempt_id}/{shot.stage}-{index}.png"
        storage_provider.upload_bytes(shot.content, key)
        await attempt_repo.add_screenshot(
            attempt_id=attempt_id, stage=shot.stage, s3_bucket=storage_provider.bucket, s3_key=key
        )
    for entry in result.logs:
        await attempt_repo.add_log_entry(
            attempt_id=attempt_id, level=entry.level, message=entry.message, context=entry.context
        )


async def _handle_requires_review(
    *,
    task_repo: AutoApplyTaskRepository,
    attempt_repo: AutoApplyAttemptRepository,
    task_id: uuid.UUID,
    attempt_id: uuid.UUID,
    task,
    attempt,
    missing_fields: list[ApplicationField],
    issues: list[ValidationIssue],
    mapped_values: dict[str, str],
) -> None:
    missing_payload = [field.model_dump() for field in missing_fields] + [
        {"field_key": issue.field_key, "label": issue.label, "message": issue.message} for issue in issues
    ]
    await task_repo.update(
        task,
        status=AutoApplyTaskStatus.REQUIRES_REVIEW,
        missing_fields=missing_payload,
        field_mapping_snapshot=mapped_values,
        error_message="Missing or invalid required fields — update your Auto Apply profile and retry.",
    )
    await attempt_repo.update(
        attempt,
        status=AutoApplyAttemptStatus.FAILED,
        failure_category=FailureCategory.VALIDATION,
        error_message="Missing or invalid required fields.",
        finished_at=datetime.now(UTC),
    )
    for field in missing_fields:
        await attempt_repo.add_log_entry(
            attempt_id=attempt_id,
            level="error",
            message=f"Missing required field: {field.label}",
            context={"key": field.key},
        )
    for issue in issues:
        await attempt_repo.add_log_entry(
            attempt_id=attempt_id, level="error", message=issue.message, context={"field_key": issue.field_key}
        )
    logger.warning(
        "Auto-apply task requires manual review",
        extra={"extra_fields": {"task_id": str(task_id), "missing_count": len(missing_payload)}},
    )


def _resume_filename(content_type: str) -> str:
    extension = "pdf" if content_type == "application/pdf" else "docx"
    return f"resume.{extension}"


async def _create_tracking_application(
    session, *, user_id: uuid.UUID, job_id: uuid.UUID | None, company_name: str, job_title: str, job_url: str
) -> uuid.UUID:
    application_repo = ApplicationRepository(session)
    history_repo = ApplicationStatusHistoryRepository(session)
    application: Application = await application_repo.create(
        user_id=user_id,
        job_id=job_id,
        company_name=company_name,
        job_title=job_title,
        job_url=job_url,
        status=ApplicationStatus.APPLIED,
        source=ApplicationSource.AUTO_APPLY,
        applied_at=datetime.now(UTC),
    )
    await history_repo.create(
        application_id=application.id, from_status=None, to_status=ApplicationStatus.APPLIED, note=None
    )
    return application.id


async def _submit_application(task_id: uuid.UUID, *, retry_count: int) -> None:
    async with AsyncSessionLocal() as session:
        task_repo = AutoApplyTaskRepository(session)
        profile_repo = ApplicantProfileRepository(session)
        attempt_repo = AutoApplyAttemptRepository(session)

        task = await task_repo.get_for_worker(task_id=task_id)
        if task is None:
            logger.warning(
                "Auto-apply task not found", extra={"extra_fields": {"task_id": str(task_id)}}
            )
            return
        if task.status == AutoApplyTaskStatus.CANCELLED:
            logger.info(
                "Auto-apply task was cancelled — skipping", extra={"extra_fields": {"task_id": str(task_id)}}
            )
            return

        # Snapshot everything this run needs as plain values *before* any
        # possible `session.rollback()` — rolled-back instances expire
        # their non-PK attributes, and this is an async session, so a
        # later implicit attribute read would raise rather than lazy-load.
        user_id = task.user_id
        job_id = task.job_id
        job_url = task.job_url
        platform = task.platform
        company_name = task.company_name
        job_title = task.job_title
        ats_resume_override = task.ats_resume_id
        resume_version_override = task.resume_version_id

        await task_repo.update(
            task, status=AutoApplyTaskStatus.RUNNING, attempt_count=retry_count + 1, error_message=None
        )
        attempt = await attempt_repo.create(
            task_id=task.id,
            attempt_number=retry_count + 1,
            status=AutoApplyAttemptStatus.RUNNING,
            started_at=datetime.now(UTC),
        )
        await session.commit()
        attempt_id = attempt.id

        storage_provider = get_container().resolve("storage_provider")

        try:
            job = await task_repo.get_job_with_company(job_id=job_id) if job_id else None
            profile = await profile_repo.get_or_create(user_id=user_id)

            ats_resumes = await task_repo.list_ats_resumes_for_user(user_id=user_id)
            resumes = await task_repo.list_resumes_for_user(user_id=user_id)
            selection = resume_selection.select_resume(
                ats_resumes=ats_resumes,
                resumes=resumes,
                job=job,
                explicit_ats_resume_id=ats_resume_override,
                explicit_resume_version_id=resume_version_override,
            )
            resume = ResumeAttachment(
                filename=_resume_filename(selection.content_type),
                content_type=selection.content_type,
                content=storage_provider.download_bytes(selection.s3_key),
            )

            intelligence = job.company.intelligence if job and job.company else None
            adapter_config = intelligence.adapter_config if intelligence else {}
            ats_identifier = intelligence.ats_identifier if intelligence else None
            external_id = job.source_job_id if job else ""

            if platform in _HTTP_PLATFORMS:
                async with build_http_client() as http_client:
                    adapter = get_apply_adapter(
                        platform,
                        job_url=job_url,
                        external_id=external_id,
                        ats_identifier=ats_identifier,
                        adapter_config=adapter_config,
                        http_client=http_client,
                    )
                    fields = await adapter.discover_fields()
                    mapping_result, issues = _map_and_validate(fields, profile)
                    if not mapping_result.is_complete or issues:
                        await _handle_requires_review(
                            task_repo=task_repo,
                            attempt_repo=attempt_repo,
                            task_id=task_id,
                            attempt_id=attempt_id,
                            task=task,
                            attempt=attempt,
                            missing_fields=mapping_result.missing_fields,
                            issues=issues,
                            mapped_values=mapping_result.mapped_values,
                        )
                        await session.commit()
                        return
                    result = await adapter.apply(field_values=mapping_result.mapped_values, resume=resume)
            else:
                adapter = get_apply_adapter(
                    platform,
                    job_url=job_url,
                    external_id=external_id,
                    ats_identifier=ats_identifier,
                    adapter_config=adapter_config,
                    http_client=None,
                )
                fields = await adapter.discover_fields()
                mapping_result, issues = _map_and_validate(fields, profile)
                if not mapping_result.is_complete or issues:
                    await _handle_requires_review(
                        task_repo=task_repo,
                        attempt_repo=attempt_repo,
                        task_id=task_id,
                        attempt_id=attempt_id,
                        task=task,
                        attempt=attempt,
                        missing_fields=mapping_result.missing_fields,
                        issues=issues,
                        mapped_values=mapping_result.mapped_values,
                    )
                    await session.commit()
                    return
                result = await adapter.apply(field_values=mapping_result.mapped_values, resume=resume)

            await _persist_apply_result(
                storage_provider=storage_provider,
                attempt_repo=attempt_repo,
                attempt_id=attempt_id,
                user_id=user_id,
                task_id=task_id,
                result=result,
            )

            if result.missing_fields:
                await _handle_requires_review(
                    task_repo=task_repo,
                    attempt_repo=attempt_repo,
                    task_id=task_id,
                    attempt_id=attempt_id,
                    task=task,
                    attempt=attempt,
                    missing_fields=result.missing_fields,
                    issues=[],
                    mapped_values=mapping_result.mapped_values,
                )
                await session.commit()
                return

            if not result.success:
                raise PlatformApplyError(result.error_message or "Application submission failed.")

            await task_repo.update(
                task,
                status=AutoApplyTaskStatus.SUBMITTED,
                resume_used_display_name=selection.display_name,
                confirmation_detected=result.confirmation_detected,
                confirmation_text=result.confirmation_text,
                confirmation_url=result.confirmation_url,
                submitted_at=datetime.now(UTC),
                field_mapping_snapshot=mapping_result.mapped_values,
                missing_fields=[],
                error_message=None,
            )
            await attempt_repo.update(
                attempt, status=AutoApplyAttemptStatus.SUCCESS, finished_at=datetime.now(UTC)
            )
            application_id = await _create_tracking_application(
                session,
                user_id=user_id,
                job_id=job_id,
                company_name=company_name,
                job_title=job_title,
                job_url=job_url,
            )
            await task_repo.update(task, tracking_application_id=application_id)
            await session.commit()
            logger.info(
                "Auto-apply submission succeeded", extra={"extra_fields": {"task_id": str(task_id)}}
            )

        except Exception as exc:
            await session.rollback()
            category = classify_exception(exc)
            is_last_attempt = retry_count >= _settings.AUTO_APPLY_MAX_ATTEMPTS - 1
            will_retry = is_retryable(category) and not is_last_attempt

            await task_repo.update(
                task,
                status=AutoApplyTaskStatus.RETRYING if will_retry else AutoApplyTaskStatus.FAILED,
                error_message=str(exc)[:2000],
            )
            await attempt_repo.update(
                attempt,
                status=AutoApplyAttemptStatus.FAILED,
                failure_category=category,
                error_message=str(exc)[:2000],
                finished_at=datetime.now(UTC),
            )
            await session.commit()
            logger.error(
                "Auto-apply submission failed",
                extra={
                    "extra_fields": {
                        "task_id": str(task_id),
                        "category": category.value,
                        "will_retry": will_retry,
                        "error": str(exc),
                    }
                },
            )
            if will_retry:
                raise TransientApplyError(str(exc)) from exc


__all__ = ["submit_application_task"]
