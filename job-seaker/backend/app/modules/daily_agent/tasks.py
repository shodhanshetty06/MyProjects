"""Daily AI Agent — Celery tasks (queue: "matching", reuses that worker
since every step here is either a DB read or delegates the heavy lifting
— AI optimization, Playwright submission — to that module's own tasks).

Every individual capability this orchestrates already existed and already
ran on its own schedule (job scanning, matching, ATS optimization, Gmail
interview-email detection); nothing here duplicates that logic. What was
missing was the daily chain connecting "strong match" -> "tailored resume"
-> "staged application" — see `ARCHITECTURE_AUDIT.md`'s documented dead
links between `matching`, `ats_resume_generator`, and `auto_apply`.

Safety gate (per product decision): this module NEVER calls
`AutoApplyService.create_task` or otherwise auto-submits an application.
It only ever calls `stage_pending_task`, which leaves the task in
`PENDING` — a human must call `POST /auto-apply/tasks/{id}/approve` before
anything is actually submitted to a real job posting.
"""

from __future__ import annotations

import uuid
from asyncio import run as run_async
from datetime import UTC, datetime

from app.core.config import get_settings
from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal
from app.dependency_injection.container import get_container
from app.modules.ats_resume_generator.models import ATSVersionStatus
from app.modules.ats_resume_generator.repository import (
    ATSResumeRepository,
    ATSResumeVersionRepository,
)
from app.modules.ats_resume_generator.schemas import ATSResumeCreateRequest
from app.modules.ats_resume_generator.service import ATSResumeGeneratorService
from app.modules.auto_apply.repository import (
    ApplicantProfileRepository,
    AutoApplyAttemptRepository,
    AutoApplyTaskRepository,
)
from app.modules.auto_apply.service import AutoApplyService
from app.modules.daily_agent.models import DailyAgentRunStatus
from app.modules.daily_agent.repository import DailyAgentRunRepository
from app.modules.matching.repository import MatchingRepository
from app.workers.base_task import BaseTask
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(base=BaseTask, bind=True, name="daily_agent.run_daily_scan")
def run_daily_scan_task(self) -> None:  # noqa: ANN001 -- Celery `self`
    run_async(_run_daily_scan())


@celery_app.task(base=BaseTask, bind=True, name="daily_agent.run_for_user")
def run_for_user_task(self, user_id: str) -> None:  # noqa: ANN001
    run_async(_run_for_user(uuid.UUID(user_id)))


@celery_app.task(
    base=BaseTask,
    bind=True,
    name="daily_agent.stage_pending_application",
    max_retries=10,
)
def stage_pending_application_task(
    self,  # noqa: ANN001
    user_id: str,
    job_id: str,
    ats_resume_id: str,
    ats_resume_version_id: str,
) -> None:
    run_async(
        _stage_pending_application(
            user_id=uuid.UUID(user_id),
            job_id=uuid.UUID(job_id),
            ats_resume_id=uuid.UUID(ats_resume_id),
            ats_resume_version_id=uuid.UUID(ats_resume_version_id),
            retries=self.request.retries,
            max_retries=self.max_retries,
        )
    )


async def _run_daily_scan() -> None:
    """Beat-triggered entry point — fans out one `run_for_user_task` per
    active user, same candidate set `matching.tasks._periodic_batch_sweep`
    already uses, so "who is the agent working for" stays consistent with
    the rest of the matching pipeline.
    """
    async with AsyncSessionLocal() as session:
        repo = MatchingRepository(session)
        user_ids = await repo.active_preference_user_ids()

    for user_id in user_ids:
        run_for_user_task.delay(str(user_id))

    logger.info("Daily agent scan fanned out", extra={"extra_fields": {"user_count": len(user_ids)}})


async def _run_for_user(user_id: uuid.UUID) -> None:
    settings = get_settings()
    started_at = datetime.now(UTC)

    async with AsyncSessionLocal() as session:
        run_repo = DailyAgentRunRepository(session)
        run = await run_repo.create(
            user_id=user_id, status=DailyAgentRunStatus.RUNNING, started_at=started_at
        )
        await session.commit()
        run_id = run.id

    jobs_scanned = 0
    matches_found = 0
    ats_generated = 0
    error_message: str | None = None

    try:
        async with AsyncSessionLocal() as session:
            matching_repo = MatchingRepository(session)
            matches = await matching_repo.list_matches_for_user(
                user_id=user_id, limit=settings.MATCHING_CANDIDATE_LIMIT
            )
            jobs_scanned = len(matches)
            qualifying = [
                m
                for m in matches
                if (float(m.match_score) if m.match_score is not None else 0.0)
                >= settings.DAILY_AGENT_ATS_GENERATION_THRESHOLD
            ]
            matches_found = len(qualifying)

            resume_version = await matching_repo.get_current_resume_version_for_user(
                user_id=user_id
            )
            if resume_version is None or resume_version.embedding is None:
                qualifying = []

            if qualifying:
                ats_repo = ATSResumeRepository(session)
                ats_service = ATSResumeGeneratorService(
                    ats_resume_repository=ats_repo,
                    version_repository=ATSResumeVersionRepository(session),
                    storage_provider=get_container().resolve("storage_provider"),
                )

                for match in qualifying:
                    job = await matching_repo.get_job(job_id=match.job_id)
                    if job is None:
                        continue
                    if await ats_repo.exists_for_target_job(user_id=user_id, job_id=job.id):
                        # Already prepared on a previous run — don't
                        # regenerate, but a stage attempt may still be
                        # worth retrying if it failed before; skipped here
                        # for simplicity (v1: one ATS resume per job is
                        # also one staging attempt per job).
                        continue

                    try:
                        ats_resume, version = await ats_service.create_ats_resume(
                            user_id=user_id,
                            payload=ATSResumeCreateRequest(
                                resume_version_id=resume_version.id,
                                job_description=job.description or job.title,
                                title=f"{job.title} — Daily Agent",
                                job_title=job.title,
                                company_name=job.company.name if job.company else None,
                                target_job_id=job.id,
                            ),
                        )
                    except (ConflictError, NotFoundError, ValidationError) as exc:
                        logger.warning(
                            "Daily agent skipped one job's ATS generation",
                            extra={
                                "extra_fields": {
                                    "user_id": str(user_id),
                                    "job_id": str(job.id),
                                    "error": str(exc),
                                }
                            },
                        )
                        continue

                    ats_generated += 1
                    await session.commit()

                    stage_pending_application_task.apply_async(
                        args=[
                            str(user_id),
                            str(job.id),
                            str(ats_resume.id),
                            str(version.id),
                        ],
                        countdown=settings.DAILY_AGENT_STAGE_CHECK_DELAY_SECONDS,
                    )

        status = DailyAgentRunStatus.COMPLETED
    except Exception as exc:  # noqa: BLE001 -- run-level failure, recorded not re-raised
        status = DailyAgentRunStatus.FAILED
        error_message = str(exc)[:2000]
        logger.error(
            "Daily agent run failed",
            extra={"extra_fields": {"user_id": str(user_id), "error": str(exc)}},
        )

    async with AsyncSessionLocal() as session:
        run_repo = DailyAgentRunRepository(session)
        run = await run_repo.get(run_id)
        if run is not None:
            await run_repo.update(
                run,
                status=status,
                completed_at=datetime.now(UTC),
                jobs_scanned=jobs_scanned,
                matches_found=matches_found,
                ats_resumes_generated=ats_generated,
                error_message=error_message,
            )
            await session.commit()

    from app.events.bus import event_bus
    from app.events.domain import DashboardUpdateRequested

    await event_bus.publish(
        DashboardUpdateRequested(
            user_id=user_id,
            reason="daily_agent_run",
            payload={
                "jobs_scanned": jobs_scanned,
                "matches_found": matches_found,
                "ats_resumes_generated": ats_generated,
                "status": status.value,
            },
        )
    )
    logger.info(
        "Daily agent run finished",
        extra={
            "extra_fields": {
                "user_id": str(user_id),
                "jobs_scanned": jobs_scanned,
                "matches_found": matches_found,
                "ats_resumes_generated": ats_generated,
                "status": status.value,
            }
        },
    )


async def _stage_pending_application(
    *,
    user_id: uuid.UUID,
    job_id: uuid.UUID,
    ats_resume_id: uuid.UUID,
    ats_resume_version_id: uuid.UUID,
    retries: int,
    max_retries: int,
) -> None:
    """Polls the ATS resume version this run just triggered until it
    reaches `COMPLETED` (Claude optimization + DOCX/PDF generation both
    finish asynchronously on their own queues), then stages a `PENDING`
    Auto-Apply task. Re-schedules itself (bounded by `max_retries`,
    Celery's own `countdown`-based retry, not a busy-loop) rather than
    blocking a worker thread waiting.
    """
    settings = get_settings()

    async with AsyncSessionLocal() as session:
        version_repo = ATSResumeVersionRepository(session)
        version = await version_repo.get_by_id(version_id=ats_resume_version_id)

        if version is None:
            logger.warning(
                "ATS resume version disappeared before staging",
                extra={"extra_fields": {"version_id": str(ats_resume_version_id)}},
            )
            return

        if version.status == ATSVersionStatus.FAILED:
            logger.warning(
                "ATS resume generation failed — daily agent will not stage this application",
                extra={"extra_fields": {"job_id": str(job_id), "user_id": str(user_id)}},
            )
            return

        if version.status != ATSVersionStatus.COMPLETED:
            if retries >= max_retries:
                logger.warning(
                    "ATS resume still not completed after max retries — giving up staging",
                    extra={"extra_fields": {"job_id": str(job_id), "user_id": str(user_id)}},
                )
                return
            stage_pending_application_task.apply_async(
                args=[str(user_id), str(job_id), str(ats_resume_id), str(ats_resume_version_id)],
                countdown=settings.DAILY_AGENT_STAGE_CHECK_DELAY_SECONDS,
                retries=retries + 1,
            )
            return

        auto_apply_service = AutoApplyService(
            profile_repository=ApplicantProfileRepository(session),
            task_repository=AutoApplyTaskRepository(session),
            attempt_repository=AutoApplyAttemptRepository(session),
            storage_provider=get_container().resolve("storage_provider"),
        )
        try:
            task = await auto_apply_service.stage_pending_task(
                user_id=user_id,
                job_id=job_id,
                ats_resume_id=ats_resume_id,
                resume_version_id=None,
            )
        except (NotFoundError, ConflictError) as exc:
            logger.warning(
                "Daily agent could not stage application",
                extra={"extra_fields": {"job_id": str(job_id), "error": str(exc)}},
            )
            return
        await session.commit()

        run_repo = DailyAgentRunRepository(session)
        latest_run = await run_repo.get_latest_for_user(user_id=user_id)
        if latest_run is not None:
            await run_repo.update(
                latest_run, applications_staged=latest_run.applications_staged + 1
            )
            await session.commit()

    from app.modules.workspace import realtime as workspace_realtime

    await workspace_realtime.publish_workspace_event(
        user_id,
        {
            "type": "auto_apply_task_staged",
            "task_id": str(task.id),
            "job_id": str(job_id),
        },
    )
    logger.info(
        "Daily agent staged an application for approval",
        extra={"extra_fields": {"user_id": str(user_id), "job_id": str(job_id)}},
    )


__all__ = ["run_daily_scan_task", "run_for_user_task", "stage_pending_application_task"]
