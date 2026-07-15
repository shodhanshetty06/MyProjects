"""Recommendation service skeleton.

Intended responsibility (future phases): rank and surface job
recommendations for a candidate (and candidate recommendations for a job)
using embedding similarity plus behavioral signals. No ranking logic is
implemented in Phase 0.
"""
from __future__ import annotations

from app.core.logging import get_logger

logger = get_logger(__name__)


class RecommendationService:
    """Coordinates job/candidate recommendation ranking."""

    def __init__(self) -> None:
        pass

    async def recommend_jobs_for_user(self, *, user_id: str, limit: int = 20):
        """Return recommended jobs for a user. Not implemented in Phase 0."""
        raise NotImplementedError(
            "RecommendationService.recommend_jobs_for_user is not implemented in Phase 0."
        )
