"""ai career coach tables

Revision ID: 0010_ai_career_coach_tables
Revises: 0009_notification_tables
Create Date: 2026-07-14
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0010_ai_career_coach_tables"
down_revision: Union[str, None] = "0009_notification_tables"
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
        "career_insight",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("insight_type", sa.String(length=40), nullable=False),
        sa.Column(
            "resume_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("resume_version.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("input_params", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("result", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("model_used", sa.String(length=100), nullable=False),
        *_timestamp_columns(),
    )
    op.create_index("ix_career_insight_user_id", "career_insight", ["user_id"])
    op.create_index("ix_career_insight_user_type", "career_insight", ["user_id", "insight_type"])

    op.create_table(
        "coach_conversation",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "title", sa.String(length=255), nullable=False, server_default="New conversation"
        ),
        sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamp_columns(),
    )
    op.create_index("ix_coach_conversation_user_id", "coach_conversation", ["user_id"])

    op.create_table(
        "coach_message",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("coach_conversation.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(length=10), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        *_timestamp_columns(),
    )
    op.create_index("ix_coach_message_conversation_id", "coach_message", ["conversation_id"])


def downgrade() -> None:
    op.drop_index("ix_coach_message_conversation_id", table_name="coach_message")
    op.drop_table("coach_message")

    op.drop_index("ix_coach_conversation_user_id", table_name="coach_conversation")
    op.drop_table("coach_conversation")

    op.drop_index("ix_career_insight_user_type", table_name="career_insight")
    op.drop_index("ix_career_insight_user_id", table_name="career_insight")
    op.drop_table("career_insight")
