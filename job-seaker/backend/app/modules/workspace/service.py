"""Workspace module — application/use-case layer.

Pure read-composition over other modules' already-existing
repositories/services (the same "owns no tables" pattern `AnalyticsService`
uses) plus a small rules-based "next best actions" list. Deliberately does
no AI calls of its own so `GET /workspace` stays cheap and synchronous.
"""
from __future__ import annotations

import uuid

from app.modules.ai_career_coach.models import CareerInsightType
from app.modules.ai_career_coach.repository import CareerInsightRepository
from app.modules.ai_career_coach.schemas import CareerInsightSummary
from app.modules.matching.models import RecommendationType
from app.modules.matching.repository import MatchingRepository
from app.modules.matching.schemas import RecommendationRead
from app.modules.matching.service import MatchingService
from app.modules.notification.repository import NotificationRepository
from app.modules.notification.schemas import NotificationRead
from app.modules.resume_parser.repository import (
    ResumePipelineRunRepository,
    ResumeVersionRepository,
)
from app.modules.resume_parser.schemas import ResumeVersionSummary
from app.modules.tracking.schemas import ApplicationRead, DashboardResponse
from app.modules.tracking.service import TrackingService
from app.modules.user_profile.repository import UserProfileRepository
from app.modules.workspace.schemas import NextBestAction, PipelineStatus, ResumeHealth, WorkspaceSummary

_UNREAD_NOTIFICATION_PREVIEW_LIMIT = 5
_ATS_BASELINE_LOW_SCORE = 70.0


class WorkspaceService:
    def __init__(
        self,
        *,
        matching_repository: MatchingRepository,
        resume_version_repository: ResumeVersionRepository,
        pipeline_run_repository: ResumePipelineRunRepository,
        user_profile_repository: UserProfileRepository,
        tracking_service: TrackingService,
        career_insight_repository: CareerInsightRepository,
        notification_repository: NotificationRepository,
    ) -> None:
        self.matching_repository = matching_repository
        self.resume_version_repository = resume_version_repository
        self.pipeline_run_repository = pipeline_run_repository
        self.user_profile_repository = user_profile_repository
        self.tracking_service = tracking_service
        self.career_insight_repository = career_insight_repository
        self.notification_repository = notification_repository
        self.matching_service = MatchingService(matching_repository)

    async def get_summary(self, *, user_id: uuid.UUID) -> WorkspaceSummary:
        profile = await self.user_profile_repository.get_by_user(user_id=user_id)
        onboarding_completed = profile is not None and profile.onboarding_completed_at is not None

        current_version = await self.matching_repository.get_current_resume_version_for_user(
            user_id=user_id
        )
        resume_health = ResumeHealth()
        resume_versions: list[ResumeVersionSummary] = []
        pipeline_status: PipelineStatus | None = None

        if current_version is not None:
            career_health_insight = await self.career_insight_repository.get_latest(
                user_id=user_id,
                insight_type=CareerInsightType.JOB_READINESS_SCORE,
                resume_version_id=current_version.id,
            )
            resume_health = ResumeHealth(
                resume_id=current_version.resume_id,
                resume_version_id=current_version.id,
                resume_score=current_version.resume_score,
                resume_score_breakdown=current_version.resume_score_breakdown,
                ats_baseline_score=current_version.ats_baseline_score,
                ats_baseline_missing_skills=current_version.ats_baseline_missing_skills,
                health_computed_at=current_version.health_computed_at,
                career_health_score=(
                    career_health_insight.result.get("overall_score")
                    if career_health_insight is not None
                    else None
                ),
            )
            resume_versions = [
                ResumeVersionSummary.model_validate(v)
                for v in await self.resume_version_repository.list_for_resume(
                    resume_id=current_version.resume_id
                )
            ]
            run = await self.pipeline_run_repository.get_by_resume_version(
                resume_version_id=current_version.id
            )
            if run is not None:
                pipeline_status = PipelineStatus(
                    run_id=run.id,
                    resume_version_id=run.resume_version_id,
                    status=run.status,
                    current_step=run.current_step,
                    progress_pct=run.progress_pct,
                    steps=run.steps,
                    error_message=run.error_message,
                )

        recommendation_dicts = await self.matching_service.list_recommendations(
            user_id=user_id, recommendation_type=RecommendationType.JOB
        )
        recommended_jobs = [RecommendationRead(**r) for r in recommendation_dicts]

        applications_statistics, recent_applications, needs_follow_up = (
            await self.tracking_service.get_dashboard(user_id=user_id)
        )
        applications = DashboardResponse(
            statistics=applications_statistics,
            recent_applications=[
                ApplicationRead.model_validate(a) for a in recent_applications
            ],
            needs_follow_up=[ApplicationRead.model_validate(a) for a in needs_follow_up],
        )

        insights = await self.career_insight_repository.list_by_user(
            user_id=user_id, insight_type=None, offset=0, limit=3
        )
        coach_highlights = [CareerInsightSummary.model_validate(i) for i in insights]

        unread_notifications = await self.notification_repository.list_filtered(
            user_id=user_id, unread_only=True, offset=0, limit=_UNREAD_NOTIFICATION_PREVIEW_LIMIT
        )
        unread_count = await self.notification_repository.count_filtered(
            user_id=user_id, unread_only=True
        )
        notifications = [NotificationRead.model_validate(n) for n in unread_notifications]

        next_best_actions = _build_next_actions(
            has_resume=current_version is not None,
            ats_baseline_score=resume_health.ats_baseline_score,
            missing_skills=resume_health.ats_baseline_missing_skills or [],
            recommended_jobs_count=len(recommended_jobs),
            needs_follow_up_count=len(needs_follow_up),
        )

        return WorkspaceSummary(
            onboarding_completed=onboarding_completed,
            resume_health=resume_health,
            recommended_jobs=recommended_jobs,
            resume_versions=resume_versions,
            applications=applications,
            coach_highlights=coach_highlights,
            notifications=notifications,
            unread_notification_count=unread_count,
            next_best_actions=next_best_actions,
            pipeline_status=pipeline_status,
        )

    async def get_pipeline_progress(self, *, user_id: uuid.UUID) -> PipelineStatus | None:
        run = await self.pipeline_run_repository.get_latest_for_user(user_id=user_id)
        if run is None:
            return None
        return PipelineStatus(
            run_id=run.id,
            resume_version_id=run.resume_version_id,
            status=run.status,
            current_step=run.current_step,
            progress_pct=run.progress_pct,
            steps=run.steps,
            error_message=run.error_message,
        )


def _build_next_actions(
    *,
    has_resume: bool,
    ats_baseline_score: float | None,
    missing_skills: list[str],
    recommended_jobs_count: int,
    needs_follow_up_count: int,
) -> list[NextBestAction]:
    actions: list[NextBestAction] = []

    if not has_resume:
        actions.append(
            NextBestAction(
                key="upload_resume",
                title="Upload your resume",
                description="Upload a resume to unlock scoring, matching, and recommendations.",
                cta_path="/resumes",
                priority=100,
            )
        )
    if ats_baseline_score is not None and ats_baseline_score < _ATS_BASELINE_LOW_SCORE:
        actions.append(
            NextBestAction(
                key="generate_ats_resume",
                title="Generate a tailored ATS resume",
                description=(
                    f"Your baseline ATS score is {ats_baseline_score:.0f} — a tailored version "
                    "for a specific job description scores meaningfully higher."
                ),
                cta_path="/resumes/ats-generator",
                priority=90,
            )
        )
    if missing_skills:
        actions.append(
            NextBestAction(
                key="close_skill_gaps",
                title="Close your top skill gaps",
                description=(
                    f"{len(missing_skills)} in-demand skill(s) for your target roles are "
                    "missing from your resume."
                ),
                cta_path="/coach/tools/skill-gap",
                priority=70,
            )
        )
    if needs_follow_up_count > 0:
        actions.append(
            NextBestAction(
                key="follow_up_applications",
                title="Follow up on stalled applications",
                description=f"{needs_follow_up_count} application(s) have had no activity in a while.",
                cta_path="/tracker",
                priority=60,
            )
        )
    if recommended_jobs_count > 0:
        actions.append(
            NextBestAction(
                key="review_recommendations",
                title="Review your recommended jobs",
                description=f"{recommended_jobs_count} job(s) matched to your profile are ready to review.",
                cta_path="/jobs/recommended",
                priority=50,
            )
        )
    if not actions:
        actions.append(
            NextBestAction(
                key="check_coach",
                title="Check in with your AI Career Coach",
                description="Get a fresh readiness check or explore a new career tool.",
                cta_path="/coach",
                priority=10,
            )
        )

    actions.sort(key=lambda a: a.priority, reverse=True)
    return actions[:4]


__all__ = ["WorkspaceService"]
