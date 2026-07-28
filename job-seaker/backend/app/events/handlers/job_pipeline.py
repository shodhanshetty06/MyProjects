"""Event-bus handlers for the New-Job pipeline domain events
(`app/events/domain.py`).

Each handler is a thin async function whose body is a 1:1 behavioral copy
of the `.delay()` call it replaces — this activation of `event_bus` changes
*coupling* (a task publishes an event instead of importing the next task
directly), not the underlying Celery execution mechanism, queues, or retry
behavior. Two handlers are strictly additive (not replacements of existing
behavior): `_on_job_embedded_refresh_recommendations` and
`_on_job_matched_update_dashboard`, closing gaps noted in the Event
Pipeline plan (new-job matches never triggered a Recommendation refresh or
reached the Workspace dashboard channel).
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal
from app.events.base import Event
from app.events.domain import DashboardUpdateRequested, JobEmbedded, JobMatched, JobNormalized
from app.events.models import PipelineEventLog

logger = get_logger(__name__)


async def _log_event(event: Event) -> None:
    """Catch-all audit-trail subscriber — writes one row per published
    event so pipeline activity (including handler-visible failures logged
    elsewhere) is queryable, not just log-line-only. `entity_id` picks
    whichever identifying field the concrete event subclass carries
    (`job_id` for job-pipeline events, `user_id` otherwise) — events don't
    share a common "subject id" field, so this is a best-effort pick, not a
    strict contract.
    """
    entity_id = getattr(event, "job_id", None) or getattr(event, "user_id", None)
    extra = event.model_dump(mode="json", exclude={"event_id", "name", "occurred_at"})
    async with AsyncSessionLocal() as session:
        session.add(
            PipelineEventLog(
                event_name=event.name,
                entity_id=entity_id,
                status="ok",
                payload=extra,
            )
        )
        await session.commit()


async def _on_job_normalized(event: Event) -> None:
    from app.modules.job_ingestion.tasks import extract_job_skills_task

    assert isinstance(event, JobNormalized)
    extract_job_skills_task.delay(str(event.job_id))


async def _on_job_embedded_match_active_users(event: Event) -> None:
    from app.modules.matching.tasks import match_new_job_for_active_users_task

    assert isinstance(event, JobEmbedded)
    match_new_job_for_active_users_task.delay(str(event.job_id))


async def _on_job_matched_refresh_recommendations(event: Event) -> None:
    """Additive: a new-job match never used to trigger a Recommendation
    refresh for the notified user — only the resume-upload chain and the
    nightly batch sweep did. Closes that gap.
    """
    from app.modules.matching.tasks import refresh_recommendations_task

    assert isinstance(event, JobMatched)
    refresh_recommendations_task.delay(str(event.user_id))


async def _on_job_matched_update_dashboard(event: Event) -> None:
    """Additive: publishes to the Workspace dashboard's own realtime
    channel (not just `search_realtime`) so a `/workspace` tab live-updates
    on a new matching job the same way it already does for the resume
    pipeline.

    Composes `DashboardUpdateRequested`'s handler directly (a plain function
    call) rather than publishing the event on the shared `event_bus`
    singleton — publishing here would only reach subscribers registered on
    whichever bus instance happens to be the *global* one, which breaks
    silently for any caller (tests included) using a different `EventBus`
    instance. `_log_event` is called alongside for the same reason: this
    function's caller already publishes `JobMatched` itself, so `.publish()`
    is reserved for genuinely new events entering the bus, not for a
    handler forwarding its own derived event to itself.
    """
    assert isinstance(event, JobMatched)
    dashboard_event = DashboardUpdateRequested(
        user_id=event.user_id,
        reason="new_matching_job",
        job_id=event.job_id,
        payload={"match_score": event.match_score},
    )
    await _on_dashboard_update_requested(dashboard_event)
    await _log_event(dashboard_event)


async def _on_dashboard_update_requested(event: Event) -> None:
    from app.modules.workspace import realtime as workspace_realtime

    assert isinstance(event, DashboardUpdateRequested)
    await workspace_realtime.publish_workspace_event(
        event.user_id,
        {
            "type": "dashboard_update",
            "reason": event.reason,
            "job_id": str(event.job_id) if event.job_id else None,
            **event.payload,
        },
    )


def register(event_bus) -> None:  # noqa: ANN001 -- avoids importing EventBus just for the hint
    event_bus.subscribe("job.normalized", _on_job_normalized)
    event_bus.subscribe("job.normalized", _log_event)
    event_bus.subscribe("job.embedded", _on_job_embedded_match_active_users)
    event_bus.subscribe("job.embedded", _log_event)
    event_bus.subscribe("job.matched", _on_job_matched_refresh_recommendations)
    event_bus.subscribe("job.matched", _on_job_matched_update_dashboard)
    event_bus.subscribe("job.matched", _log_event)
    event_bus.subscribe("recommendations.refreshed", _log_event)
    event_bus.subscribe("notification.sent", _log_event)
    event_bus.subscribe("dashboard.update_requested", _on_dashboard_update_requested)
    event_bus.subscribe("dashboard.update_requested", _log_event)


__all__ = ["register"]
