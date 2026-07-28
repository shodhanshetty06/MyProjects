"""workspace onboarding tables

Revision ID: 0013_workspace_onboarding_tables
Revises: 0012_auth_tables
Create Date: 2026-07-23
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0013_workspace_onboarding_tables"
down_revision: Union[str, None] = "0012_auth_tables"
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
    op.create_table(
        "user_profile",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False, unique=True),
        sa.Column("target_roles", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("work_authorization", sa.String(length=100), nullable=True),
        sa.Column("onboarding_completed_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamp_columns(),
    )
    op.create_index("ix_user_profile_user_id", "user_profile", ["user_id"])

    op.create_table(
        "resume_pipeline_run",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "resume_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("resume_version.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("current_step", sa.String(length=50), nullable=True),
        sa.Column("progress_pct", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("steps", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("error_message", sa.Text(), nullable=True),
        *_timestamp_columns(),
    )
    op.create_index("ix_resume_pipeline_run_user_id", "resume_pipeline_run", ["user_id"])

    op.add_column("resume_version", sa.Column("resume_score", sa.Float(), nullable=True))
    op.add_column(
        "resume_version", sa.Column("resume_score_breakdown", postgresql.JSONB(), nullable=True)
    )
    op.add_column("resume_version", sa.Column("ats_baseline_score", sa.Float(), nullable=True))
    op.add_column(
        "resume_version", sa.Column("ats_baseline_missing_skills", postgresql.JSONB(), nullable=True)
    )
    op.add_column(
        "resume_version",
        sa.Column("health_computed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("resume_version", "health_computed_at")
    op.drop_column("resume_version", "ats_baseline_missing_skills")
    op.drop_column("resume_version", "ats_baseline_score")
    op.drop_column("resume_version", "resume_score_breakdown")
    op.drop_column("resume_version", "resume_score")

    op.drop_index("ix_resume_pipeline_run_user_id", table_name="resume_pipeline_run")
    op.drop_table("resume_pipeline_run")

    op.drop_index("ix_user_profile_user_id", table_name="user_profile")
    op.drop_table("user_profile")
