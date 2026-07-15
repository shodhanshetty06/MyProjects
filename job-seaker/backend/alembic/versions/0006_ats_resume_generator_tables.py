"""ats resume generator tables

Revision ID: 0006_ats_resume_generator_tables
Revises: 0005_tracking_tables
Create Date: 2026-07-13
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0006_ats_resume_generator_tables"
down_revision: Union[str, None] = "0005_tracking_tables"
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
        "ats_resume",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "source_resume_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("resume.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "target_job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("job.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("job_title", sa.String(length=255), nullable=True),
        sa.Column("company_name", sa.String(length=255), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamp_columns(),
    )
    op.create_index("ix_ats_resume_user_id", "ats_resume", ["user_id"])
    op.create_index("ix_ats_resume_source_resume_id", "ats_resume", ["source_resume_id"])

    op.create_table(
        "ats_resume_version",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "ats_resume_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ats_resume.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column(
            "source_resume_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("resume_version.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("job_description", sa.Text(), nullable=False, server_default=""),
        sa.Column("experience", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("education", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("certifications", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("languages", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("achievements", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("original_summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("optimized_summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("original_skills", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("optimized_skills", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("original_projects", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("optimized_projects", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("keywords", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("matched_keywords", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("missing_keywords", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("keyword_score", sa.Numeric(5, 2), nullable=True),
        sa.Column("ats_score", sa.Numeric(5, 2), nullable=True),
        sa.Column("score_breakdown", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("model_used", sa.String(length=100), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamp_columns(),
        sa.UniqueConstraint("ats_resume_id", "version_number", name="uq_ats_resume_version_number"),
    )
    op.create_index("ix_ats_resume_version_ats_resume_id", "ats_resume_version", ["ats_resume_id"])

    op.create_table(
        "ats_resume_file",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "ats_resume_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ats_resume_version.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("format", sa.String(length=10), nullable=False),
        sa.Column("s3_bucket", sa.String(length=255), nullable=False),
        sa.Column("s3_key", sa.String(length=1024), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        *_timestamp_columns(),
        sa.UniqueConstraint(
            "ats_resume_version_id", "format", name="uq_ats_resume_file_format"
        ),
    )
    op.create_index(
        "ix_ats_resume_file_ats_resume_version_id", "ats_resume_file", ["ats_resume_version_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_ats_resume_file_ats_resume_version_id", table_name="ats_resume_file")
    op.drop_table("ats_resume_file")

    op.drop_index("ix_ats_resume_version_ats_resume_id", table_name="ats_resume_version")
    op.drop_table("ats_resume_version")

    op.drop_index("ix_ats_resume_source_resume_id", table_name="ats_resume")
    op.drop_index("ix_ats_resume_user_id", table_name="ats_resume")
    op.drop_table("ats_resume")
