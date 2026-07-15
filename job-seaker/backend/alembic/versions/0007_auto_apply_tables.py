"""auto apply tables

Revision ID: 0007_auto_apply_tables
Revises: 0006_ats_resume_generator_tables
Create Date: 2026-07-13
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0007_auto_apply_tables"
down_revision: Union[str, None] = "0006_ats_resume_generator_tables"
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
        "applicant_profile",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False, unique=True),
        sa.Column("auto_apply_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("first_name", sa.String(length=100), nullable=True),
        sa.Column("last_name", sa.String(length=100), nullable=True),
        sa.Column("email", sa.String(length=255), nullable=True),
        sa.Column("phone", sa.String(length=50), nullable=True),
        sa.Column("address_line1", sa.String(length=255), nullable=True),
        sa.Column("city", sa.String(length=255), nullable=True),
        sa.Column("state", sa.String(length=255), nullable=True),
        sa.Column("postal_code", sa.String(length=20), nullable=True),
        sa.Column("country", sa.String(length=255), nullable=True),
        sa.Column("linkedin_url", sa.String(length=512), nullable=True),
        sa.Column("portfolio_url", sa.String(length=512), nullable=True),
        sa.Column("github_url", sa.String(length=512), nullable=True),
        sa.Column("website_url", sa.String(length=512), nullable=True),
        sa.Column("work_authorized", sa.Boolean(), nullable=True),
        sa.Column("requires_sponsorship", sa.Boolean(), nullable=True),
        sa.Column("willing_to_relocate", sa.Boolean(), nullable=True),
        sa.Column("years_of_experience", sa.Integer(), nullable=True),
        sa.Column("desired_salary", sa.String(length=100), nullable=True),
        sa.Column("notice_period", sa.String(length=100), nullable=True),
        sa.Column("gender", sa.String(length=100), nullable=True),
        sa.Column("race_ethnicity", sa.String(length=100), nullable=True),
        sa.Column("veteran_status", sa.String(length=100), nullable=True),
        sa.Column("disability_status", sa.String(length=100), nullable=True),
        sa.Column(
            "default_ats_resume_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ats_resume.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "default_resume_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("resume.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("custom_answers", postgresql.JSONB(), nullable=False, server_default="{}"),
        *_timestamp_columns(),
    )
    op.create_index("ix_applicant_profile_user_id", "applicant_profile", ["user_id"], unique=True)

    op.create_table(
        "auto_apply_task",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("job.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("company_name", sa.String(length=255), nullable=False),
        sa.Column("job_title", sa.String(length=512), nullable=False),
        sa.Column("job_url", sa.String(length=1024), nullable=False),
        sa.Column("platform", sa.String(length=20), nullable=False),
        sa.Column(
            "ats_resume_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ats_resume.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "resume_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("resume_version.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("resume_used_display_name", sa.String(length=512), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("field_mapping_snapshot", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("missing_fields", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("confirmation_detected", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("confirmation_text", sa.Text(), nullable=True),
        sa.Column("confirmation_url", sa.String(length=1024), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "tracking_application_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("application.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamp_columns(),
    )
    op.create_index("ix_auto_apply_task_user_id", "auto_apply_task", ["user_id"])
    op.create_index("ix_auto_apply_task_job_id", "auto_apply_task", ["job_id"])
    op.create_index("ix_auto_apply_task_status", "auto_apply_task", ["status"])

    op.create_table(
        "auto_apply_attempt",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "task_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("auto_apply_task.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="running"),
        sa.Column("failure_category", sa.String(length=20), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        *_timestamp_columns(),
    )
    op.create_index("ix_auto_apply_attempt_task_id", "auto_apply_attempt", ["task_id"])

    op.create_table(
        "auto_apply_screenshot",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "attempt_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("auto_apply_attempt.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stage", sa.String(length=20), nullable=False),
        sa.Column("s3_bucket", sa.String(length=255), nullable=False),
        sa.Column("s3_key", sa.String(length=1024), nullable=False),
        *_timestamp_columns(),
    )
    op.create_index("ix_auto_apply_screenshot_attempt_id", "auto_apply_screenshot", ["attempt_id"])

    op.create_table(
        "auto_apply_log_entry",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "attempt_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("auto_apply_attempt.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("level", sa.String(length=10), nullable=False, server_default="info"),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("context", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_auto_apply_log_entry_attempt_id", "auto_apply_log_entry", ["attempt_id"])


def downgrade() -> None:
    op.drop_index("ix_auto_apply_log_entry_attempt_id", table_name="auto_apply_log_entry")
    op.drop_table("auto_apply_log_entry")

    op.drop_index("ix_auto_apply_screenshot_attempt_id", table_name="auto_apply_screenshot")
    op.drop_table("auto_apply_screenshot")

    op.drop_index("ix_auto_apply_attempt_task_id", table_name="auto_apply_attempt")
    op.drop_table("auto_apply_attempt")

    op.drop_index("ix_auto_apply_task_status", table_name="auto_apply_task")
    op.drop_index("ix_auto_apply_task_job_id", table_name="auto_apply_task")
    op.drop_index("ix_auto_apply_task_user_id", table_name="auto_apply_task")
    op.drop_table("auto_apply_task")

    op.drop_index("ix_applicant_profile_user_id", table_name="applicant_profile")
    op.drop_table("applicant_profile")
