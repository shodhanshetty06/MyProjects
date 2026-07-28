"""job ingestion: company intelligence profile, AI job understanding, quality score

Adds the columns backing three job-ingestion features:
  - Company Intelligence profile: company_size, funding_stage, remote_policy,
    tech_stack on `company_intelligence`.
  - AI Job Understanding: required_skills, preferred_skills, responsibilities,
    benefits, tech_stack, seniority, industry, department on `job`.
  - Job Quality Score: quality_score, quality_reasons on `job`.

Revision ID: 0017_job_intel_quality
Revises: 0016_search_saved_job_table
Create Date: 2026-07-27
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0017_job_intel_quality"
down_revision: Union[str, None] = "0016_search_saved_job_table"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- Company Intelligence profile (Feature 4) ---
    op.add_column(
        "company_intelligence", sa.Column("company_size", sa.String(length=50), nullable=True)
    )
    op.add_column(
        "company_intelligence", sa.Column("funding_stage", sa.String(length=50), nullable=True)
    )
    op.add_column(
        "company_intelligence",
        sa.Column(
            "remote_policy", sa.String(length=20), nullable=False, server_default="unknown"
        ),
    )
    op.add_column(
        "company_intelligence",
        sa.Column("tech_stack", postgresql.JSONB(), nullable=False, server_default="[]"),
    )

    # --- AI Job Understanding (Feature 3) ---
    op.add_column(
        "job", sa.Column("required_skills", postgresql.JSONB(), nullable=False, server_default="[]")
    )
    op.add_column(
        "job", sa.Column("preferred_skills", postgresql.JSONB(), nullable=False, server_default="[]")
    )
    op.add_column(
        "job", sa.Column("responsibilities", postgresql.JSONB(), nullable=False, server_default="[]")
    )
    op.add_column(
        "job", sa.Column("benefits", postgresql.JSONB(), nullable=False, server_default="[]")
    )
    op.add_column(
        "job", sa.Column("tech_stack", postgresql.JSONB(), nullable=False, server_default="[]")
    )
    op.add_column(
        "job",
        sa.Column("seniority", sa.String(length=20), nullable=False, server_default="unknown"),
    )
    op.add_column("job", sa.Column("industry", sa.String(length=255), nullable=True))
    op.add_column("job", sa.Column("department", sa.String(length=255), nullable=True))

    # --- Job Quality Score (Feature 5) ---
    op.add_column("job", sa.Column("quality_score", sa.Integer(), nullable=True))
    op.add_column(
        "job", sa.Column("quality_reasons", postgresql.JSONB(), nullable=False, server_default="[]")
    )
    op.create_index("ix_job_quality_score", "job", ["quality_score"])
    op.create_index("ix_job_seniority", "job", ["seniority"])


def downgrade() -> None:
    op.drop_index("ix_job_seniority", table_name="job")
    op.drop_index("ix_job_quality_score", table_name="job")
    op.drop_column("job", "quality_reasons")
    op.drop_column("job", "quality_score")
    op.drop_column("job", "department")
    op.drop_column("job", "industry")
    op.drop_column("job", "seniority")
    op.drop_column("job", "tech_stack")
    op.drop_column("job", "benefits")
    op.drop_column("job", "responsibilities")
    op.drop_column("job", "preferred_skills")
    op.drop_column("job", "required_skills")
    op.drop_column("company_intelligence", "tech_stack")
    op.drop_column("company_intelligence", "remote_policy")
    op.drop_column("company_intelligence", "funding_stage")
    op.drop_column("company_intelligence", "company_size")
