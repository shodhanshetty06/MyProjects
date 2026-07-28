"""Integration tests for the activated event bus (`app/events/bus.py`) and
its job-pipeline handlers (`app/events/handlers/job_pipeline.py`). Uses the
real test database (see `conftest.py`) since the catch-all `_log_event`
subscriber writes a `PipelineEventLog` row — the whole point being tested is
that publishing an event both (a) triggers the expected downstream Celery
`.delay()` call and (b) leaves an audit trail, not just one or the other.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.events.bus import EventBus
from app.events.domain import DashboardUpdateRequested, JobEmbedded, JobMatched, JobNormalized
from app.events.handlers import job_pipeline
from app.events.models import PipelineEventLog


@pytest.fixture
def bus() -> EventBus:
    """A fresh bus per test (not the shared `app.events.bus.event_bus`
    singleton) so tests don't leak subscriptions into each other.
    """
    fresh = EventBus()
    job_pipeline.register(fresh)
    return fresh


@pytest.mark.asyncio
async def test_job_normalized_enqueues_skill_extraction(bus: EventBus) -> None:
    job_id = uuid.uuid4()
    with patch(
        "app.modules.job_ingestion.tasks.extract_job_skills_task.delay"
    ) as mock_delay:
        await bus.publish(JobNormalized(job_id=job_id))
    mock_delay.assert_called_once_with(str(job_id))


@pytest.mark.asyncio
async def test_job_embedded_enqueues_matching(bus: EventBus) -> None:
    job_id = uuid.uuid4()
    with patch(
        "app.modules.matching.tasks.match_new_job_for_active_users_task.delay"
    ) as mock_delay:
        await bus.publish(JobEmbedded(job_id=job_id))
    mock_delay.assert_called_once_with(str(job_id))


@pytest.mark.asyncio
async def test_job_matched_refreshes_recommendations_and_updates_dashboard(
    bus: EventBus,
) -> None:
    job_id, user_id = uuid.uuid4(), uuid.uuid4()
    with (
        patch("app.modules.matching.tasks.refresh_recommendations_task.delay") as mock_refresh,
        patch(
            "app.modules.workspace.realtime.publish_workspace_event"
        ) as mock_publish,
    ):
        await bus.publish(JobMatched(job_id=job_id, user_id=user_id, match_score=88.0))

    mock_refresh.assert_called_once_with(str(user_id))
    # JobMatched -> DashboardUpdateRequested -> workspace realtime publish,
    # closing the gap where new-job matches never reached the Workspace WS.
    mock_publish.assert_called_once()
    published_user_id, payload = mock_publish.call_args.args
    assert published_user_id == user_id
    assert payload["type"] == "dashboard_update"
    assert payload["job_id"] == str(job_id)


@pytest.mark.asyncio
async def test_every_published_event_is_audit_logged(bus: EventBus) -> None:
    from app.db.session import AsyncSessionLocal

    user_id = uuid.uuid4()
    with patch("app.modules.workspace.realtime.publish_workspace_event"):
        await bus.publish(
            DashboardUpdateRequested(user_id=user_id, reason="test_reason")
        )

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(PipelineEventLog).where(PipelineEventLog.event_name == "dashboard.update_requested")
        )
        rows = result.scalars().all()

    assert any(row.entity_id == user_id for row in rows)
