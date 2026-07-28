"""Daily AI Agent module — audit trail for autonomous daily runs.

One row per user per day the agent runs: scans that user's fresh matches,
generates ATS resumes for the strongest ones, and stages (never submits —
see `auto_apply.service.AutoApplyService.stage_pending_task`) applications
for review. Mirrors `resume_parser.ResumePipelineRun` / `job_ingestion
.IngestionRun`'s "one row records one automated run" convention.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base_class import Base, TimestampMixin, UUIDPrimaryKeyMixin


class DailyAgentRunStatus(str, enum.Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DailyAgentRun(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """No FK on `user_id` — same no-cross-module-FK convention every other
    `user_id` column in this codebase already follows (ownership enforced
    at the service layer, not the database).
    """

    __tablename__ = "daily_agent_run"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    status: Mapped[DailyAgentRunStatus] = mapped_column(
        String(20), nullable=False, default=DailyAgentRunStatus.RUNNING
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    jobs_scanned: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    matches_found: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ats_resumes_generated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    applications_staged: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    error_message: Mapped[str | None] = mapped_column(String(2000), nullable=True)


__all__ = ["DailyAgentRun", "DailyAgentRunStatus"]
