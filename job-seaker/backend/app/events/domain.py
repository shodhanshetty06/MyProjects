"""Domain events for the automatic New-Job pipeline (Normalize -> Validate
-> Embedding -> Match -> Recommendation -> Notification -> Dashboard
Update).

Every stage of this pipeline already ran automatically before this file
existed — via hardcoded `task.delay()` calls chained at the end of each
Celery task body. These events don't change *that* execution mechanism
(each handler in `app/events/handlers/job_pipeline.py` still just calls the
same `.delay()`), they replace the *coupling*: a task now publishes "I
finished normalizing" instead of importing and calling the skill-extraction
task directly, so a new reaction to a pipeline stage can be added by
registering a new subscriber instead of editing the upstream module.
"""

from __future__ import annotations

import uuid

from app.events.base import Event


class JobNormalized(Event):
    """A job posting finished normalization (salary/location/work-mode
    parsed, status set to ACTIVE) and is ready for AI skill extraction.
    """

    name: str = "job.normalized"
    job_id: uuid.UUID


class JobEmbedded(Event):
    """A job posting's embedding vector has been generated and persisted —
    ready to be matched against active users' resumes.
    """

    name: str = "job.embedded"
    job_id: uuid.UUID


class JobMatched(Event):
    """One user's resume was matched against a newly-ingested job and
    cleared the notification threshold. Carries enough to drive both the
    Recommendation-refresh and Dashboard-update reactions without either
    handler re-querying the match.
    """

    name: str = "job.matched"
    job_id: uuid.UUID
    user_id: uuid.UUID
    match_score: float


class RecommendationsRefreshed(Event):
    """A user's `Recommendation` rows were recomputed."""

    name: str = "recommendations.refreshed"
    user_id: uuid.UUID


class NotificationSent(Event):
    """A notification was delivered to a user — published purely for
    observability/future subscribers (e.g. the Daily Agent's run log); the
    actual send already happened by the time this is published.
    """

    name: str = "notification.sent"
    user_id: uuid.UUID
    event_type: str


class DashboardUpdateRequested(Event):
    """Something changed that the user's Workspace dashboard should reflect
    without a manual page refresh — the terminal stage of every pipeline
    variant (new-job match, resume pipeline, daily agent run, ...).
    """

    name: str = "dashboard.update_requested"
    user_id: uuid.UUID
    reason: str
    job_id: uuid.UUID | None = None


__all__ = [
    "JobNormalized",
    "JobEmbedded",
    "JobMatched",
    "RecommendationsRefreshed",
    "NotificationSent",
    "DashboardUpdateRequested",
]
