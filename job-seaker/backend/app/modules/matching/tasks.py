"""Matching module — Celery tasks (queue: "matching").

Sync-Celery-wrapping-`asyncio.run` convention, same as
`resume_parser/tasks.py`: each task is a thin wrapper so the existing async
DB session (`AsyncSessionLocal`) can be reused without a parallel sync
SQLAlchemy stack.
"""

from __future__ import annotations

import uuid
from asyncio import run as run_async
from datetime import UTC, datetime, timedelta

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal
from app.modules.matching import realtime
from app.modules.matching.repository import MatchingRepository
from app.modules.matching.service import MatchingService
from app.workers.base_task import BaseTask
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


async def _advance_pipeline(
    *, user_id: uuid.UUID, resume_version_id: uuid.UUID, step: str, status: str, error: str | None = None
) -> None:
    """Same helper as `resume_parser.tasks._advance_pipeline` — duplicated
    rather than imported to avoid a resume_parser<->matching import cycle
    (matching already imports resume_parser models, not the other way).
    """
    from app.modules.resume_parser.repository import ResumePipelineRunRepository
    from app.modules.workspace import realtime as workspace_realtime

    async with AsyncSessionLocal() as session:
        repo = ResumePipelineRunRepository(session)
        run = await repo.advance(
            resume_version_id=resume_version_id, step=step, status=status, error=error
        )
        await session.commit()

    if run is None:
        return

    await workspace_realtime.publish_workspace_event(
        user_id,
        {
            "type": "pipeline_progress",
            "run_id": str(run.id),
            "resume_version_id": str(resume_version_id),
            "step": step,
            "step_status": status,
            "status": run.status.value,
            "progress_pct": run.progress_pct,
        },
    )


@celery_app.task(base=BaseTask, bind=True, name="matching.compute_resume_matches")
def compute_resume_matches_task(
    self, resume_version_id: str
) -> None:  # noqa: ANN001 -- Celery `self`
    run_async(_compute_resume_matches(uuid.UUID(resume_version_id)))


@celery_app.task(base=BaseTask, bind=True, name="matching.refresh_recommendations")
def refresh_recommendations_task(
    self, user_id: str, resume_version_id: str | None = None
) -> None:  # noqa: ANN001
    run_async(
        _refresh_recommendations(
            uuid.UUID(user_id),
            uuid.UUID(resume_version_id) if resume_version_id else None,
        )
    )


@celery_app.task(base=BaseTask, bind=True, name="matching.sweep_stale_matches")
def sweep_stale_matches_task(self) -> None:  # noqa: ANN001
    run_async(_sweep_stale_matches())


@celery_app.task(base=BaseTask, bind=True, name="matching.periodic_batch_sweep")
def periodic_batch_sweep_task(self) -> None:  # noqa: ANN001
    run_async(_periodic_batch_sweep())


@celery_app.task(
    base=BaseTask,
    bind=True,
    name="matching.match_new_job_for_active_users",
    autoretry_for=(Exception,),
    max_retries=3,
    retry_backoff=True,
)
def match_new_job_for_active_users_task(self, job_id: str) -> None:  # noqa: ANN001
    run_async(_match_new_job_for_active_users(uuid.UUID(job_id)))


async def _compute_resume_matches(resume_version_id: uuid.UUID) -> None:
    async with AsyncSessionLocal() as session:
        repo = MatchingRepository(session)
        resume_version = await repo.get_resume_version(resume_version_id=resume_version_id)
        if resume_version is None or resume_version.resume is None:
            logger.warning(
                "Resume version not found for match computation",
                extra={"extra_fields": {"resume_version_id": str(resume_version_id)}},
            )
            return
        if resume_version.embedding is None:
            logger.warning(
                "Resume version has no embedding yet, skipping match computation",
                extra={"extra_fields": {"resume_version_id": str(resume_version_id)}},
            )
            return

        user_id = resume_version.resume.user_id
        await _advance_pipeline(
            user_id=user_id,
            resume_version_id=resume_version_id,
            step="matching",
            status="running",
        )

        service = MatchingService(repo)
        count = await service.batch_compute_matches(
            user_id=user_id, resume_version_id=resume_version_id
        )
        await session.commit()

    await realtime.publish_match_event(
        user_id,
        {
            "type": "matches_computed",
            "resume_version_id": str(resume_version_id),
            "count": count,
        },
    )
    logger.info(
        "Resume matches computed",
        extra={"extra_fields": {"resume_version_id": str(resume_version_id), "count": count}},
    )
    await _advance_pipeline(
        user_id=user_id,
        resume_version_id=resume_version_id,
        step="matching",
        status="completed",
    )
    refresh_recommendations_task.delay(str(user_id), str(resume_version_id))


async def _refresh_recommendations(
    user_id: uuid.UUID, resume_version_id: uuid.UUID | None
) -> None:
    if resume_version_id is not None:
        await _advance_pipeline(
            user_id=user_id,
            resume_version_id=resume_version_id,
            step="recommendations",
            status="running",
        )

    async with AsyncSessionLocal() as session:
        repo = MatchingRepository(session)
        service = MatchingService(repo)
        result = await service.refresh_recommendations(user_id=user_id)
        await session.commit()

    logger.info(
        "Recommendations refreshed",
        extra={"extra_fields": {"user_id": str(user_id), **result.model_dump()}},
    )

    if resume_version_id is not None:
        await _advance_pipeline(
            user_id=user_id,
            resume_version_id=resume_version_id,
            step="recommendations",
            status="completed",
        )
        from app.modules.ai_career_coach.tasks import compute_dashboard_insights_task

        compute_dashboard_insights_task.delay(str(resume_version_id), str(user_id))
    else:
        from app.modules.workspace import realtime as workspace_realtime

        await workspace_realtime.publish_workspace_event(
            user_id, {"type": "recommendations_updated", **result.model_dump()}
        )


async def _sweep_stale_matches() -> None:
    settings = get_settings()
    cutoff = datetime.now(UTC) - timedelta(hours=settings.MATCHING_STALE_MATCH_HOURS)
    async with AsyncSessionLocal() as session:
        repo = MatchingRepository(session)
        count = await repo.mark_stale_before(cutoff=cutoff)
        await session.commit()

    logger.info(
        "Stale matches swept",
        extra={"extra_fields": {"count": count, "cutoff": cutoff.isoformat()}},
    )


async def _periodic_batch_sweep() -> None:
    async with AsyncSessionLocal() as session:
        repo = MatchingRepository(session)
        user_ids = await repo.active_preference_user_ids()
        version_ids: list[uuid.UUID] = []
        for user_id in user_ids:
            resume_version = await repo.get_current_resume_version_for_user(user_id=user_id)
            if resume_version is not None and resume_version.embedding is not None:
                version_ids.append(resume_version.id)

    for version_id in version_ids:
        compute_resume_matches_task.delay(str(version_id))

    logger.info(
        "Periodic batch sweep fanned out",
        extra={"extra_fields": {"count": len(version_ids)}},
    )


async def _match_new_job_for_active_users(job_id: uuid.UUID) -> None:
    """Smart Trigger: New Job Ingested -> match against affected users ->
    notify users whose score clears `MATCHING_NEW_JOB_NOTIFY_THRESHOLD`.

    Reuses `MatchingService.compute_match` (the same single-pair path the
    `POST /matching/resumes/{id}/jobs/{id}/compute` endpoint uses) per
    active-preference user with a current, embedded resume — the same
    candidate set `_periodic_batch_sweep` iterates, just triggered
    immediately for one job instead of on a schedule.

    Also publishes a `new_matching_jobs` event on the `search:updates:*`
    channel (see `app.modules.search.realtime`) for every notified user —
    the "refresh search" leg of "new jobs arrive -> recalculate match ->
    update recommendations -> refresh search -> notify -> update workspace"
    (plan section 7). This is the only place `search` and `matching` touch;
    no scoring logic is duplicated, just one extra publish call alongside
    the existing notification.
    """
    from app.modules.auth.repository import AuthRepository
    from app.modules.notification.models import NotificationEventType
    from app.modules.notification.repository import (
        NotificationPreferenceRepository,
        NotificationRepository,
    )
    from app.modules.notification.service import NotificationService
    from app.modules.search import realtime as search_realtime

    settings = get_settings()
    notified = 0

    async with AsyncSessionLocal() as session:
        repo = MatchingRepository(session)
        job = await repo.get_job(job_id=job_id)
        if job is None:
            logger.warning(
                "Job not found for new-job match fan-out",
                extra={"extra_fields": {"job_id": str(job_id)}},
            )
            return

        service = MatchingService(repo)
        notification_service = NotificationService(
            repository=NotificationRepository(session),
            preference_repository=NotificationPreferenceRepository(session),
        )
        auth_repo = AuthRepository(session)

        user_ids = await repo.active_preference_user_ids()
        for user_id in user_ids:
            resume_version = await repo.get_current_resume_version_for_user(user_id=user_id)
            if resume_version is None or resume_version.embedding is None:
                continue

            try:
                match = await service.compute_match(
                    user_id=user_id, resume_version_id=resume_version.id, job_id=job_id
                )
            except Exception as exc:
                logger.warning(
                    "Skipping new-job match for one user",
                    extra={"extra_fields": {"user_id": str(user_id), "error": str(exc)}},
                )
                continue

            if (match.match_score or 0) >= settings.MATCHING_NEW_JOB_NOTIFY_THRESHOLD:
                user = await auth_repo.get(user_id)
                await notification_service.notify(
                    user_id=user_id,
                    event_type=NotificationEventType.NEW_MATCHING_JOB,
                    title=f"New match: {job.title} at {job.company.name}",
                    body=(
                        f"{job.title} at {job.company.name} scored "
                        f"{match.match_score:.0f}% for your profile."
                    ),
                    data={"job_id": str(job_id), "match_score": match.match_score},
                    recipient_email=user.email if user else None,
                )
                from app.events.bus import event_bus
                from app.events.domain import NotificationSent

                await event_bus.publish(
                    NotificationSent(
                        user_id=user_id, event_type=NotificationEventType.NEW_MATCHING_JOB.value
                    )
                )
                await search_realtime.publish_search_event(
                    user_id,
                    {
                        "type": "new_matching_jobs",
                        "job_id": str(job_id),
                        "match_score": match.match_score,
                    },
                )
                # Publishes JobMatched on the event bus (see
                # `app/events/handlers/job_pipeline.py`) — additive over the
                # notify+search_realtime calls above: it triggers a
                # Recommendation refresh for this user (previously only the
                # resume-upload chain / nightly sweep did that) and a
                # Workspace-dashboard update (previously only
                # `search_realtime` was reached from this path, not the
                # `/workspace` WS channel).
                from app.events.bus import event_bus
                from app.events.domain import JobMatched

                await event_bus.publish(
                    JobMatched(
                        job_id=job_id, user_id=user_id, match_score=match.match_score or 0.0
                    )
                )
                notified += 1
        await session.commit()

    logger.info(
        "New job matched against active users",
        extra={"extra_fields": {"job_id": str(job_id), "notified": notified}},
    )
