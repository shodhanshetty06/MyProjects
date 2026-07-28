"""Workspace module — Celery tasks (queue: "matching", reusing the existing
worker rather than standing up a dedicated one for one lightweight task).

`update_workspace_task` is the second-to-last link in the auto-orchestrated
"ResumeUploaded" pipeline (see `app.modules.resume_parser.repository.
PIPELINE_STEPS`): it marks the run COMPLETED and hands off to
`notification.tasks.send_resume_ready_notification_task`.
"""
from __future__ import annotations

import uuid
from asyncio import run as run_async

from app.core.logging import get_logger
from app.workers.base_task import BaseTask
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(
    base=BaseTask,
    bind=True,
    name="workspace.update_dashboard",
    autoretry_for=(Exception,),
    max_retries=3,
    retry_backoff=True,
)
def update_workspace_task(self, user_id: str, resume_version_id: str) -> None:  # noqa: ANN001
    run_async(_update_workspace(uuid.UUID(user_id), uuid.UUID(resume_version_id)))


async def _update_workspace(user_id: uuid.UUID, resume_version_id: uuid.UUID) -> None:
    from app.modules.resume_parser.tasks import _advance_pipeline

    await _advance_pipeline(
        user_id=user_id, resume_version_id=resume_version_id, step="dashboard", status="running"
    )

    from app.modules.workspace import realtime as workspace_realtime

    await workspace_realtime.publish_workspace_event(
        user_id,
        {"type": "workspace_ready", "resume_version_id": str(resume_version_id)},
    )

    await _advance_pipeline(
        user_id=user_id, resume_version_id=resume_version_id, step="dashboard", status="completed"
    )

    logger.info(
        "Workspace updated",
        extra={"extra_fields": {"user_id": str(user_id), "resume_version_id": str(resume_version_id)}},
    )

    from app.modules.notification.tasks import send_resume_ready_notification_task

    send_resume_ready_notification_task.delay(str(user_id), str(resume_version_id))
