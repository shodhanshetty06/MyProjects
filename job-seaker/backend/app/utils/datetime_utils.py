"""Shared datetime helpers used across modules."""
from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Timezone-aware current UTC timestamp.

    Prefer this over `datetime.utcnow()` (naive, deprecated) everywhere in
    the codebase for consistency with `TimestampMixin`'s server defaults.
    """
    return datetime.now(timezone.utc)
