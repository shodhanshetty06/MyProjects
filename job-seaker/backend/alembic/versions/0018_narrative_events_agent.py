"""AI Explanation narrative column, Event Pipeline audit log, Daily AI Agent run log

Adds the schema backing three of the five plan features:
  - AI Explanation: resume_job_match.ai_narrative (cached LLM one-liner).
  - Event Pipeline: pipeline_event_log table (queryable event-bus audit trail).
  - Daily AI Agent: daily_agent_run table (one row per user per agent run).

Feature 2 (Better Search NL parsing) and Feature 4 (Ranking Engine) add no
schema — both are pure computation over existing columns.

Revision ID: 0018_narrative_events_agent
Revises: 0017_job_intel_quality
Create Date: 2026-07-27
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0018_narrative_events_agent"
down_revision: Union[str, None] = "0017_job_intel_quality"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _timestamp_columns() -> list[sa.Column]:
    return [
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
    ]


def upgrade() -> None:
    # --- AI Explanation ---
    op.add_column("resume_job_match", sa.Column("ai_narrative", sa.Text(), nullable=True))

    # --- Event Pipeline audit trail ---
    op.create_table(
        "pipeline_event_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("event_name", sa.String(length=100), nullable=False),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="ok"),
        sa.Column("payload", postgresql.JSONB(), nullable=False, server_default="{}"),
        *_timestamp_columns(),
    )
    op.create_index("ix_pipeline_event_log_event_name", "pipeline_event_log", ["event_name"])
    op.create_index("ix_pipeline_event_log_entity_id", "pipeline_event_log", ["entity_id"])

    # --- Daily AI Agent run log ---
    op.create_table(
        "daily_agent_run",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="running"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("jobs_scanned", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("matches_found", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("ats_resumes_generated", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("applications_staged", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.String(length=2000), nullable=True),
        *_timestamp_columns(),
    )
    op.create_index("ix_daily_agent_run_user_id", "daily_agent_run", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_daily_agent_run_user_id", table_name="daily_agent_run")
    op.drop_table("daily_agent_run")
    op.drop_index("ix_pipeline_event_log_entity_id", table_name="pipeline_event_log")
    op.drop_index("ix_pipeline_event_log_event_name", table_name="pipeline_event_log")
    op.drop_table("pipeline_event_log")
    op.drop_column("resume_job_match", "ai_narrative")
