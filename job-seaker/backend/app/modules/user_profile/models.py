"""User Profile module — ORM models.

`UserProfile` holds onboarding-only fields that don't belong in
`matching.MatchPreference` (which already owns everything preference-shaped:
countries, work modes, employment types, salary, years of experience).
`user_id` has no foreign key, mirroring `MatchPreference`'s convention —
ownership is enforced at the service layer against the JWT-derived caller id.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base_class import Base, TimestampMixin, UUIDPrimaryKeyMixin


class UserProfile(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "user_profile"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, unique=True, index=True
    )
    target_roles: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    work_authorization: Mapped[str | None] = mapped_column(String(100), nullable=True)
    onboarding_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


__all__ = ["UserProfile"]
