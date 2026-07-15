"""User Profile module — ORM models.

Phase 0: no tables are defined yet. Models for this module will subclass
`Base`, `UUIDPrimaryKeyMixin`, and `TimestampMixin` (see
`app.db.base_class`) once the User Profile domain is implemented, e.g.:

    class UserProfile(Base, UUIDPrimaryKeyMixin, TimestampMixin):
        __tablename__ = "user_profile"
"""
from app.db.base_class import Base  # noqa: F401
