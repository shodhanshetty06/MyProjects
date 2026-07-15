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


@celery_app.task(base=BaseTask, bind=True, name="matching.compute_resume_matches")
def compute_resume_matches_task(
    self, resume_version_id: str
) -> None:  # noqa: ANN001 -- Celery `self`
    run_async(_compute_resume_matches(uuid.UUID(resume_version_id)))


@celery_app.task(base=BaseTask, bind=True, name="matching.refresh_recommendations")
def refresh_recommendations_task(self, user_id: str) -> None:  # noqa: ANN001
    run_async(_refresh_recommendations(uuid.UUID(user_id)))


@celery_app.task(base=BaseTask, bind=True, name="matching.sweep_stale_matches")
def sweep_stale_matches_task(self) -> None:  # noqa: ANN001
    run_async(_sweep_stale_matches())


@celery_app.task(base=BaseTask, bind=True, name="matching.periodic_batch_sweep")
def periodic_batch_sweep_task(self) -> None:  # noqa: ANN001
    run_async(_periodic_batch_sweep())


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
    refresh_recommendations_task.delay(str(user_id))


async def _refresh_recommendations(user_id: uuid.UUID) -> None:
    async with AsyncSessionLocal() as session:
        repo = MatchingRepository(session)
        service = MatchingService(repo)
        result = await service.refresh_recommendations(user_id=user_id)
        await session.commit()

    logger.info(
        "Recommendations refreshed",
        extra={"extra_fields": {"user_id": str(user_id), **result.model_dump()}},
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
