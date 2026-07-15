"""job ingestion tables

Revision ID: 0002_job_ingestion
Revises: 0001_resume_parser
Create Date: 2026-07-11
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0002_job_ingestion"
down_revision: Union[str, None] = "0001_resume_parser"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

VECTOR_DIMENSIONS = 1024


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
        "company",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("normalized_name", sa.String(length=255), nullable=False, unique=True),
        sa.Column("website", sa.String(length=512), nullable=True),
        sa.Column("logo_url", sa.String(length=1024), nullable=True),
        sa.Column("industry", sa.String(length=255), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        *_timestamp_columns(),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_company_normalized_name", "company", ["normalized_name"])

    op.create_table(
        "company_intelligence",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "company_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("company.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("ats_platform", sa.String(length=20), nullable=False, server_default="unknown"),
        sa.Column("ats_identifier", sa.String(length=255), nullable=True),
        sa.Column("career_url", sa.String(length=1024), nullable=True),
        sa.Column("adapter_config", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("last_crawled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_crawl_status", sa.String(length=20), nullable=True),
        sa.Column("hiring_frequency_per_month", sa.Numeric(10, 2), nullable=True),
        sa.Column("company_rating", sa.Numeric(3, 2), nullable=True),
        sa.Column("hiring_trends", postgresql.JSONB(), nullable=True),
        *_timestamp_columns(),
    )

    op.create_table(
        "location",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("raw_text", sa.String(length=512), nullable=False),
        sa.Column("city", sa.String(length=255), nullable=True),
        sa.Column("region", sa.String(length=255), nullable=True),
        sa.Column("country", sa.String(length=255), nullable=True),
        sa.Column("country_code", sa.String(length=2), nullable=True),
        sa.Column("normalized_key", sa.String(length=512), nullable=False, unique=True),
        *_timestamp_columns(),
    )
    op.create_index("ix_location_normalized_key", "location", ["normalized_key"])

    op.create_table(
        "skill",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("normalized_name", sa.String(length=255), nullable=False, unique=True),
        *_timestamp_columns(),
    )
    op.create_index("ix_skill_normalized_name", "skill", ["normalized_name"])

    op.create_table(
        "job",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "company_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("company.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "location_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("location.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("source", sa.String(length=20), nullable=False),
        sa.Column("source_job_id", sa.String(length=255), nullable=False),
        sa.Column("source_url", sa.String(length=1024), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column(
            "employment_type", sa.String(length=20), nullable=False, server_default="unknown"
        ),
        sa.Column("work_mode", sa.String(length=20), nullable=False, server_default="unknown"),
        sa.Column("salary_min", sa.Numeric(12, 2), nullable=True),
        sa.Column("salary_max", sa.Numeric(12, 2), nullable=True),
        sa.Column("salary_currency", sa.String(length=3), nullable=True),
        sa.Column("salary_period", sa.String(length=20), nullable=False, server_default="unknown"),
        sa.Column("salary_raw_text", sa.String(length=255), nullable=True),
        sa.Column("posted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "last_seen_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("raw_payload", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("error_message", sa.Text(), nullable=True),
        *_timestamp_columns(),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("source", "source_job_id", name="uq_job_source_source_job_id"),
    )
    op.create_index("ix_job_company_id", "job", ["company_id"])
    op.create_index("ix_job_location_id", "job", ["location_id"])
    op.create_index("ix_job_content_hash", "job", ["content_hash"])

    op.create_table(
        "job_skill",
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("job.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "skill_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("skill.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )

    op.create_table(
        "job_embedding",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("job.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("embedding", Vector(VECTOR_DIMENSIONS), nullable=False),
        sa.Column("model_used", sa.String(length=100), nullable=False),
        *_timestamp_columns(),
    )

    op.create_table(
        "ingestion_run",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "company_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("company.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("jobs_found", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("jobs_created", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("jobs_updated", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("jobs_closed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("jobs_failed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        *_timestamp_columns(),
    )
    op.create_index("ix_ingestion_run_company_id", "ingestion_run", ["company_id"])


def downgrade() -> None:
    op.drop_index("ix_ingestion_run_company_id", table_name="ingestion_run")
    op.drop_table("ingestion_run")
    op.drop_table("job_embedding")
    op.drop_table("job_skill")
    op.drop_index("ix_job_content_hash", table_name="job")
    op.drop_index("ix_job_location_id", table_name="job")
    op.drop_index("ix_job_company_id", table_name="job")
    op.drop_table("job")
    op.drop_index("ix_skill_normalized_name", table_name="skill")
    op.drop_table("skill")
    op.drop_index("ix_location_normalized_key", table_name="location")
    op.drop_table("location")
    op.drop_table("company_intelligence")
    op.drop_index("ix_company_normalized_name", table_name="company")
    op.drop_table("company")
