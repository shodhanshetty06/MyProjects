"""Workspace module — Pydantic response contracts.

This module owns no tables (same "pure aggregation" pattern as `analytics`)
— every field here is assembled from other modules' already-existing
repositories/services in `service.py`.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel

from app.modules.ai_career_coach.schemas import CareerInsightSummary
from app.modules.matching.schemas import RecommendationRead
from app.modules.notification.schemas import NotificationRead
from app.modules.resume_parser.models import PipelineRunStatus
from app.modules.resume_parser.schemas import ResumeVersionSummary
from app.modules.tracking.schemas import DashboardResponse


class ResumeHealth(BaseModel):
    resume_id: uuid.UUID | None = None
    resume_version_id: uuid.UUID | None = None
    resume_score: float | None = None
    resume_score_breakdown: dict | None = None
    ats_baseline_score: float | None = None
    ats_baseline_missing_skills: list[str] | None = None
    health_computed_at: datetime | None = None
    career_health_score: float | None = None


class PipelineStatus(BaseModel):
    run_id: uuid.UUID
    resume_version_id: uuid.UUID
    status: PipelineRunStatus
    current_step: str | None
    progress_pct: int
    steps: list[dict]
    error_message: str | None = None


class NextBestAction(BaseModel):
    key: str
    title: str
    description: str
    cta_path: str
    priority: int = 0


class WorkspaceSummary(BaseModel):
    onboarding_completed: bool
    resume_health: ResumeHealth
    recommended_jobs: list[RecommendationRead]
    resume_versions: list[ResumeVersionSummary]
    applications: DashboardResponse
    coach_highlights: list[CareerInsightSummary]
    notifications: list[NotificationRead]
    unread_notification_count: int
    next_best_actions: list[NextBestAction]
    pipeline_status: PipelineStatus | None = None


__all__ = [
    "ResumeHealth",
    "PipelineStatus",
    "NextBestAction",
    "WorkspaceSummary",
]
