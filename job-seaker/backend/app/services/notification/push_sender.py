"""Push notification channel — explicit placeholder for this phase.

No FCM/APNs (or any real push provider) integration yet. `send` just logs
what would have been sent and returns, so the PUSH channel exists end-to-end
(preferences, `Notification` rows, call sites) without a real delivery
mechanism. Swapping this for a real provider later is a same-signature
change — no caller needs to know the difference.
"""

from __future__ import annotations

from app.core.logging import get_logger

logger = get_logger(__name__)


class PushSender:
    async def send(self, *, user_id: str, title: str, body: str) -> None:
        logger.info(
            "Push notification (placeholder — no real delivery configured)",
            extra={"extra_fields": {"user_id": user_id, "title": title}},
        )


__all__ = ["PushSender"]
