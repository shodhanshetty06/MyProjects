"""search saved_job table

Revision ID: 0016_search_saved_job_table
Revises: 0015_performance_indexes
Create Date: 2026-07-25
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0016_search_saved_job_table"
down_revision: Union[str, None] = "0015_performance_indexes"
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
        "saved_job",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("job.id", ondelete="CASCADE"),
            nullable=False,
        ),
        *_timestamp_columns(),
        sa.UniqueConstraint("user_id", "job_id", name="uq_saved_job_user_job"),
    )
    op.create_index("ix_saved_job_user_id", "saved_job", ["user_id"])
    op.create_index("ix_saved_job_job_id", "saved_job", ["job_id"])


def downgrade() -> None:
    op.drop_index("ix_saved_job_job_id", table_name="saved_job")
    op.drop_index("ix_saved_job_user_id", table_name="saved_job")
    op.drop_table("saved_job")
