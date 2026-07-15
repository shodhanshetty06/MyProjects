"""User Profile module — application/use-case layer."""
from __future__ import annotations

from app.core.logging import get_logger
from app.modules.user_profile.repository import UserProfileRepository

logger = get_logger(__name__)


class UserProfileService:
    """Business logic orchestration for the User Profile module.

    Phase 0: no use cases are implemented yet.
    """

    def __init__(self, repository: UserProfileRepository) -> None:
        self.repository = repository
