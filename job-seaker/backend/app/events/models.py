"""Queryable audit trail for domain events published on `event_bus`.

Written by a catch-all logging handler (see
`app/events/handlers/job_pipeline.py::_log_event`) subscribed to every event
name — gives pipeline-stage failures (e.g. a validation rejection during job
ingestion) a queryable home instead of being log-line-only, per the
"validation-failure visibility" gap the Event Pipeline plan called out.
"""

from __future__ import annotations

import uuid

from sqlalchemy import String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base_class import Base, TimestampMixin, UUIDPrimaryKeyMixin


class PipelineEventLog(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One row per published domain event. `entity_id` is whatever the
    event's primary subject is (job_id, user_id, ...) — untyped/unindexed
    beyond a plain UUID column since it refers to different tables
    depending on `event_name`, same "no cross-module FK" convention used
    for every `user_id` column elsewhere in this codebase.
    """

    __tablename__ = "pipeline_event_log"

    event_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    entity_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="ok")
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)


__all__ = ["PipelineEventLog"]
