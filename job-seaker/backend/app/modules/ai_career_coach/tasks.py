"""AI Career Coach module — Celery tasks (queue: "matching", reusing the
existing worker rather than standing up a dedicated one).

`compute_dashboard_insights_task` is the link between `matching.tasks.
refresh_recommendations` and `workspace.tasks.update_workspace_task` in the
auto-orchestrated "ResumeUploaded" pipeline (see `app.modules.resume_parser.
repository.PIPELINE_STEPS`): it precomputes Skill Gap Analysis, Learning
Recommendations, and Career Health (job readiness) so the workspace never
has to compute them on demand.

These three all require a target role, which only exists once
`user_profile.target_roles` is set (usually via onboarding completion) — so
this task is triggered from two places: the automatic pipeline hand-off
(target role often not known yet — the three steps are simply left pending
for now) and `POST /onboarding/complete` (target role now known — this is
where the real work usually happens).
"""
from __future__ import annotations

import uuid
from asyncio import run as run_async

from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal
from app.dependency_injection.container import get_container
from app.modules.ai_career_coach.repository import (
    CareerInsightRepository,
    CoachConversationRepository,
    CoachMessageRepository,
    JobMarketRepository,
    ResumeContextRepository,
)
from app.modules.ai_career_coach.service import AICareerCoachService
from app.modules.resume_parser.tasks import _advance_pipeline
from app.workers.base_task import BaseTask
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(base=BaseTask, bind=True, name="ai_career_coach.compute_dashboard_insights")
def compute_dashboard_insights_task(
    self, resume_version_id: str, user_id: str
) -> None:  # noqa: ANN001 -- Celery `self`
    run_async(_compute_dashboard_insights(uuid.UUID(resume_version_id), uuid.UUID(user_id)))


async def _resolve_target_role(*, session, user_id: uuid.UUID, resume_version_id: uuid.UUID) -> str | None:
    from app.modules.user_profile.repository import UserProfileRepository

    profile = await UserProfileRepository(session).get_by_user(user_id=user_id)
    if profile and profile.target_roles:
        return profile.target_roles[0]

    version = await ResumeContextRepository(session).get_owned_version(
        user_id=user_id, resume_version_id=resume_version_id
    )
    if version and version.parsed_data and version.parsed_data.experience:
        title = version.parsed_data.experience[0].get("title")
        if title and title.strip():
            return title.strip()
    return None


async def _compute_dashboard_insights(resume_version_id: uuid.UUID, user_id: uuid.UUID) -> None:
    async with AsyncSessionLocal() as session:
        target_role = await _resolve_target_role(
            session=session, user_id=user_id, resume_version_id=resume_version_id
        )

    if target_role is None:
        logger.info(
            "No target role available yet, skipping dashboard insight precompute",
            extra={"extra_fields": {"resume_version_id": str(resume_version_id), "user_id": str(user_id)}},
        )
        # Mark these steps completed rather than leaving them pending forever:
        # this task only fires twice (automatic pipeline hand-off, then once
        # more from `/onboarding/complete`), and a user can complete onboarding
        # without ever setting a target role. If these stayed "pending", the
        # `ResumePipelineRun` could never reach COMPLETED and the "Setting up
        # your workspace" screen would poll forever with no way out.
        for step in ("skill_gap", "learning_recommendations", "career_health"):
            await _advance_pipeline(
                user_id=user_id, resume_version_id=resume_version_id, step=step, status="completed"
            )
    else:
        ai_service = get_container().resolve("ai_service")

        # --- skill gap ---------------------------------------------------------------------
        await _advance_pipeline(
            user_id=user_id, resume_version_id=resume_version_id, step="skill_gap", status="running"
        )
        missing_skills: list[str] = []
        try:
            async with AsyncSessionLocal() as session:
                service = _build_service(session, ai_service)
                gap = await service.analyze_skill_gap(
                    user_id=user_id, resume_version_id=resume_version_id, target_role=target_role
                )
                await session.commit()
            missing_skills = [item.skill for item in gap.missing_skills]
        except Exception as exc:
            logger.error(
                "Skill gap precompute failed",
                extra={"extra_fields": {"resume_version_id": str(resume_version_id), "error": str(exc)}},
            )
            await _advance_pipeline(
                user_id=user_id,
                resume_version_id=resume_version_id,
                step="skill_gap",
                status="failed",
                error=str(exc)[:2000],
            )
        else:
            await _advance_pipeline(
                user_id=user_id, resume_version_id=resume_version_id, step="skill_gap", status="completed"
            )

        # --- learning recommendations -------------------------------------------------------
        await _advance_pipeline(
            user_id=user_id,
            resume_version_id=resume_version_id,
            step="learning_recommendations",
            status="running",
        )
        try:
            async with AsyncSessionLocal() as session:
                service = _build_service(session, ai_service)
                await service.recommend_learning(
                    user_id=user_id,
                    resume_version_id=resume_version_id,
                    target_role=target_role,
                    skills=missing_skills or None,
                )
                await session.commit()
        except Exception as exc:
            logger.error(
                "Learning recommendations precompute failed",
                extra={"extra_fields": {"resume_version_id": str(resume_version_id), "error": str(exc)}},
            )
            await _advance_pipeline(
                user_id=user_id,
                resume_version_id=resume_version_id,
                step="learning_recommendations",
                status="failed",
                error=str(exc)[:2000],
            )
        else:
            await _advance_pipeline(
                user_id=user_id,
                resume_version_id=resume_version_id,
                step="learning_recommendations",
                status="completed",
            )

        # --- career health (job readiness) ---------------------------------------------------
        await _advance_pipeline(
            user_id=user_id, resume_version_id=resume_version_id, step="career_health", status="running"
        )
        try:
            async with AsyncSessionLocal() as session:
                service = _build_service(session, ai_service)
                await service.compute_job_readiness(
                    user_id=user_id, resume_version_id=resume_version_id, target_role=target_role
                )
                await session.commit()
        except Exception as exc:
            logger.error(
                "Career health precompute failed",
                extra={"extra_fields": {"resume_version_id": str(resume_version_id), "error": str(exc)}},
            )
            await _advance_pipeline(
                user_id=user_id,
                resume_version_id=resume_version_id,
                step="career_health",
                status="failed",
                error=str(exc)[:2000],
            )
        else:
            await _advance_pipeline(
                user_id=user_id,
                resume_version_id=resume_version_id,
                step="career_health",
                status="completed",
            )

    from app.modules.workspace.tasks import update_workspace_task

    update_workspace_task.delay(str(user_id), str(resume_version_id))


def _build_service(session, ai_service) -> AICareerCoachService:  # noqa: ANN001
    return AICareerCoachService(
        insight_repository=CareerInsightRepository(session),
        conversation_repository=CoachConversationRepository(session),
        message_repository=CoachMessageRepository(session),
        resume_context_repository=ResumeContextRepository(session),
        job_market_repository=JobMarketRepository(session),
        ai_service=ai_service,
    )
