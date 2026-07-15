"""Re-exports of shared ORM mixins for convenient importing from `app.models`.

Actual definitions live in `app.db.base_class` (single source of truth so
Alembic's `Base.metadata` and module models resolve to the same classes).
"""
from app.db.base_class import (  # noqa: F401
    Base,
    SoftDeleteMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)
