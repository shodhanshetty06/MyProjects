"""admin tables

Revision ID: 0011_admin_tables
Revises: 0010_ai_career_coach_tables
Create Date: 2026-07-14
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0011_admin_tables"
down_revision: Union[str, None] = "0010_ai_career_coach_tables"
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
        "admin_actor",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False, unique=True),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False, server_default="support"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("last_active_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamp_columns(),
    )
    op.create_index("ix_admin_actor_user_id", "admin_actor", ["user_id"])

    op.create_table(
        "platform_user_status",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False, unique=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="active"),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "changed_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("admin_actor.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "changed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        *_timestamp_columns(),
    )
    op.create_index("ix_platform_user_status_user_id", "platform_user_status", ["user_id"])

    op.create_table(
        "company_verification",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "company_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("company.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="unverified"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "reviewed_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("admin_actor.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamp_columns(),
    )

    op.create_table(
        "job_moderation",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("job.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="active"),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "moderated_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("admin_actor.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("moderated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("flag_count", sa.Integer(), nullable=False, server_default="0"),
        *_timestamp_columns(),
    )

    op.create_table(
        "content_report",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("reporter_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("target_type", sa.String(length=20), nullable=False),
        sa.Column("target_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="open"),
        sa.Column(
            "resolved_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("admin_actor.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("resolution_notes", sa.Text(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamp_columns(),
    )
    op.create_index("ix_content_report_reporter_user_id", "content_report", ["reporter_user_id"])
    op.create_index("ix_content_report_target_id", "content_report", ["target_id"])
    op.create_index("ix_content_report_status", "content_report", ["status"])

    op.create_table(
        "audit_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "actor_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("admin_actor.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("action", sa.String(length=100), nullable=False),
        sa.Column("target_type", sa.String(length=50), nullable=False),
        sa.Column("target_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("before", postgresql.JSONB(), nullable=True),
        sa.Column("after", postgresql.JSONB(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_audit_log_actor_id", "audit_log", ["actor_id"])
    op.create_index("ix_audit_log_action", "audit_log", ["action"])
    op.create_index("ix_audit_log_target_type", "audit_log", ["target_type"])
    op.create_index("ix_audit_log_target_id", "audit_log", ["target_id"])
    op.create_index("ix_audit_log_created_at", "audit_log", ["created_at"])

    op.create_table(
        "feature_flag",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("key", sa.String(length=100), nullable=False, unique=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("rollout_percentage", sa.Integer(), nullable=False, server_default="100"),
        sa.Column(
            "updated_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("admin_actor.id", ondelete="SET NULL"),
            nullable=True,
        ),
        *_timestamp_columns(),
    )
    op.create_index("ix_feature_flag_key", "feature_flag", ["key"])


def downgrade() -> None:
    op.drop_index("ix_feature_flag_key", table_name="feature_flag")
    op.drop_table("feature_flag")

    op.drop_index("ix_audit_log_created_at", table_name="audit_log")
    op.drop_index("ix_audit_log_target_id", table_name="audit_log")
    op.drop_index("ix_audit_log_target_type", table_name="audit_log")
    op.drop_index("ix_audit_log_action", table_name="audit_log")
    op.drop_index("ix_audit_log_actor_id", table_name="audit_log")
    op.drop_table("audit_log")

    op.drop_index("ix_content_report_status", table_name="content_report")
    op.drop_index("ix_content_report_target_id", table_name="content_report")
    op.drop_index("ix_content_report_reporter_user_id", table_name="content_report")
    op.drop_table("content_report")

    op.drop_table("job_moderation")

    op.drop_table("company_verification")

    op.drop_index("ix_platform_user_status_user_id", table_name="platform_user_status")
    op.drop_table("platform_user_status")

    op.drop_index("ix_admin_actor_user_id", table_name="admin_actor")
    op.drop_table("admin_actor")
