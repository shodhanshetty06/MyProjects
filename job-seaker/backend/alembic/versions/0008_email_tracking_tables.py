"""email tracking tables

Revision ID: 0008_email_tracking_tables
Revises: 0007_auto_apply_tables
Create Date: 2026-07-13
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0008_email_tracking_tables"
down_revision: Union[str, None] = "0007_auto_apply_tables"
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
        "gmail_account",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email_address", sa.String(length=255), nullable=False, unique=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="connected"),
        sa.Column("access_token_encrypted", sa.Text(), nullable=False),
        sa.Column("refresh_token_encrypted", sa.Text(), nullable=False),
        sa.Column("token_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("scopes", sa.String(length=512), nullable=False),
        sa.Column("history_id", sa.String(length=50), nullable=True),
        sa.Column("last_synced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_sync_error", sa.Text(), nullable=True),
        *_timestamp_columns(),
    )
    op.create_index("ix_gmail_account_user_id", "gmail_account", ["user_id"])
    op.create_index("ix_gmail_account_email_address", "gmail_account", ["email_address"], unique=True)

    op.create_table(
        "tracked_email",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "gmail_account_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("gmail_account.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "application_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("application.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("gmail_message_id", sa.String(length=100), nullable=False),
        sa.Column("thread_id", sa.String(length=100), nullable=True),
        sa.Column("sender", sa.String(length=512), nullable=False),
        sa.Column("subject", sa.String(length=1024), nullable=False),
        sa.Column("snippet", sa.Text(), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("classification", sa.String(length=20), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("match_status", sa.String(length=20), nullable=False),
        sa.Column(
            "matched_status_history_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("application_status_history.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("raw_metadata", postgresql.JSONB(), nullable=False, server_default="{}"),
        *_timestamp_columns(),
        sa.UniqueConstraint(
            "gmail_account_id", "gmail_message_id", name="uq_tracked_email_account_message"
        ),
    )
    op.create_index("ix_tracked_email_gmail_account_id", "tracked_email", ["gmail_account_id"])
    op.create_index("ix_tracked_email_application_id", "tracked_email", ["application_id"])
    op.create_index("ix_tracked_email_match_status", "tracked_email", ["match_status"])


def downgrade() -> None:
    op.drop_index("ix_tracked_email_match_status", table_name="tracked_email")
    op.drop_index("ix_tracked_email_application_id", table_name="tracked_email")
    op.drop_index("ix_tracked_email_gmail_account_id", table_name="tracked_email")
    op.drop_table("tracked_email")

    op.drop_index("ix_gmail_account_email_address", table_name="gmail_account")
    op.drop_index("ix_gmail_account_user_id", table_name="gmail_account")
    op.drop_table("gmail_account")
