"""pgvector integration helpers.

pgvector runs inside the same Postgres instance (see
`backend/docker/postgres/init/01_extensions.sql`), so there is no separate
connection to manage — this module provides the shared `Vector` column
type and dimension constant so module models stay consistent.

Usage in a module's models.py (once business logic is implemented):

    from pgvector.sqlalchemy import Vector
    from app.integrations.vector_store import VECTOR_DIMENSIONS

    embedding: Mapped[list[float]] = mapped_column(Vector(VECTOR_DIMENSIONS))
"""
from __future__ import annotations

from app.core.config import get_settings

settings = get_settings()
VECTOR_DIMENSIONS: int = settings.VECTOR_DIMENSIONS
