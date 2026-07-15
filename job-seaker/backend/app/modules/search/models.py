"""Search module — ORM models.

No tables are owned by this module. Search queries `job_ingestion`'s
`Job`/`Company`/`Location`/`Skill`/`JobEmbedding` models directly; the only
schema changes this module needs (full-text `search_vector` column, GIN/
trigram/HNSW indexes) are added to those existing tables by
`alembic/versions/0003_search_indexes.py`.
"""

from app.db.base_class import Base  # noqa: F401
