"""search indexes

Revision ID: 0003_search_indexes
Revises: 0002_job_ingestion
Create Date: 2026-07-11
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003_search_indexes"
down_revision: Union[str, None] = "0002_job_ingestion"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Full-text search: generated tsvector column (title weighted above
    # description) + GIN index. `plainto_tsquery`/`ts_rank_cd` in
    # SearchRepository query against this column.
    op.execute(
        "ALTER TABLE job ADD COLUMN search_vector tsvector "
        "GENERATED ALWAYS AS ("
        "  setweight(to_tsvector('english', coalesce(title, '')), 'A') || "
        "  setweight(to_tsvector('english', coalesce(description, '')), 'B')"
        ") STORED"
    )
    op.execute("CREATE INDEX ix_job_search_vector ON job USING GIN (search_vector)")

    # Trigram indexes (pg_trgm, already enabled in
    # backend/docker/postgres/init/01_extensions.sql) for typo-tolerant
    # keyword search and ILIKE-prefix/infix autocomplete + suggestions.
    op.execute("CREATE INDEX ix_job_title_trgm ON job USING GIN (title gin_trgm_ops)")
    op.execute("CREATE INDEX ix_company_name_trgm ON company USING GIN (name gin_trgm_ops)")
    op.execute("CREATE INDEX ix_skill_name_trgm ON skill USING GIN (name gin_trgm_ops)")

    # Partial index covering the `status='active' AND deleted_at IS NULL`
    # predicate shared by nearly every search query, plus NEWEST sort /
    # date_posted filtering.
    op.execute(
        "CREATE INDEX ix_job_active_posted_at ON job (posted_at DESC) "
        "WHERE status = 'active' AND deleted_at IS NULL"
    )

    # HNSW ANN index for pgvector cosine similarity — `0002_job_ingestion`
    # only created the raw `embedding` column, no index, so semantic search
    # would otherwise fall back to an exact scan of every embedding.
    op.execute(
        "CREATE INDEX ix_job_embedding_cosine ON job_embedding "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_job_embedding_cosine")
    op.execute("DROP INDEX IF EXISTS ix_job_active_posted_at")
    op.execute("DROP INDEX IF EXISTS ix_skill_name_trgm")
    op.execute("DROP INDEX IF EXISTS ix_company_name_trgm")
    op.execute("DROP INDEX IF EXISTS ix_job_title_trgm")
    op.execute("DROP INDEX IF EXISTS ix_job_search_vector")
    op.execute("ALTER TABLE job DROP COLUMN search_vector")
