"""tracking tables

Revision ID: 0005_tracking_tables
Revises: 0004_matching_tables
Create Date: 2026-07-13
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0005_tracking_tables"
down_revision: Union[str, None] = "0004_matching_tables"
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
        "application",
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
        sa.Column("job_url", sa.String(length=1024), nullable=True),
        sa.Column("location", sa.String(length=512), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="applied"),
        sa.Column("source", sa.String(length=20), nullable=False, server_default="manual"),
        sa.Column(
            "applied_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamp_columns(),
    )
    op.create_index("ix_application_user_id", "application", ["user_id"])
    op.create_index("ix_application_job_id", "application", ["job_id"])
    op.create_index("ix_application_status", "application", ["status"])

    op.create_table(
        "application_status_history",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "application_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("application.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("from_status", sa.String(length=20), nullable=True),
        sa.Column("to_status", sa.String(length=20), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        *_timestamp_columns(),
    )
    op.create_index(
        "ix_application_status_history_application_id",
        "application_status_history",
        ["application_id"],
    )

    op.create_table(
        "application_note",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "application_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("application.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        *_timestamp_columns(),
    )
    op.create_index("ix_application_note_application_id", "application_note", ["application_id"])


def downgrade() -> None:
    op.drop_index("ix_application_note_application_id", table_name="application_note")
    op.drop_table("application_note")

    op.drop_index(
        "ix_application_status_history_application_id", table_name="application_status_history"
    )
    op.drop_table("application_status_history")

    op.drop_index("ix_application_status", table_name="application")
    op.drop_index("ix_application_job_id", table_name="application")
    op.drop_index("ix_application_user_id", table_name="application")
    op.drop_table("application")
