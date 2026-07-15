"""matching tables

Revision ID: 0004_matching_tables
Revises: 0003_search_indexes
Create Date: 2026-07-12
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0004_matching_tables"
down_revision: Union[str, None] = "0003_search_indexes"
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
        "match_preference",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False, unique=True),
        sa.Column("preferred_countries", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("preferred_work_modes", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column(
            "preferred_employment_types", postgresql.JSONB(), nullable=False, server_default="[]"
        ),
        sa.Column("years_of_experience", sa.Integer(), nullable=True),
        sa.Column("expected_salary_min", sa.Numeric(12, 2), nullable=True),
        sa.Column("expected_salary_max", sa.Numeric(12, 2), nullable=True),
        sa.Column("salary_currency", sa.String(length=3), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        *_timestamp_columns(),
    )
    op.create_index("ix_match_preference_user_id", "match_preference", ["user_id"])

    op.create_table(
        "resume_job_match",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "resume_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("resume_version.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("job.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("match_score", sa.Numeric(5, 2), nullable=True),
        sa.Column("semantic_score", sa.Numeric(6, 5), nullable=True),
        sa.Column("skill_match_score", sa.Numeric(5, 2), nullable=True),
        sa.Column("rule_boost_score", sa.Numeric(5, 2), nullable=True),
        sa.Column("matched_skills", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("missing_skills", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("strengths", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("weaknesses", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("score_breakdown", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("error_message", sa.Text(), nullable=True),
        *_timestamp_columns(),
        sa.UniqueConstraint("resume_version_id", "job_id", name="uq_resume_job_match_pair"),
    )
    op.create_index(
        "ix_resume_job_match_resume_version_id", "resume_job_match", ["resume_version_id"]
    )
    op.create_index("ix_resume_job_match_job_id", "resume_job_match", ["job_id"])
    op.create_index("ix_resume_job_match_user_id", "resume_job_match", ["user_id"])

    op.create_table(
        "recommendation",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("recommendation_type", sa.String(length=20), nullable=False),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("job.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "company_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("company.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "skill_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("skill.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("score", sa.Numeric(5, 2), nullable=False, server_default="0"),
        sa.Column("reason", sa.Text(), nullable=False, server_default=""),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="active"),
        *_timestamp_columns(),
    )
    op.create_index("ix_recommendation_user_id", "recommendation", ["user_id"])
    op.create_index(
        "ix_recommendation_user_type_status",
        "recommendation",
        ["user_id", "recommendation_type", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_recommendation_user_type_status", table_name="recommendation")
    op.drop_index("ix_recommendation_user_id", table_name="recommendation")
    op.drop_table("recommendation")

    op.drop_index("ix_resume_job_match_user_id", table_name="resume_job_match")
    op.drop_index("ix_resume_job_match_job_id", table_name="resume_job_match")
    op.drop_index("ix_resume_job_match_resume_version_id", table_name="resume_job_match")
    op.drop_table("resume_job_match")

    op.drop_index("ix_match_preference_user_id", table_name="match_preference")
    op.drop_table("match_preference")
