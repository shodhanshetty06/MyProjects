"""In-process async event bus (publish/subscribe).

Phase 0 provides the mechanism only — no domain events are published yet.
Modules register handlers against an event name; `publish()` fans the
event out to every registered handler concurrently.

For cross-process fan-out (e.g. notifying Celery workers), a handler can
simply enqueue a Celery task as its implementation — the bus itself stays
transport-agnostic.

Usage (future phase):
    event_bus.subscribe("user.registered", send_welcome_email_handler)
    await event_bus.publish(UserRegisteredEvent(user_id=user.id))
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Awaitable, Callable

from app.core.logging import get_logger
from app.events.base import Event

logger = get_logger(__name__)

EventHandler = Callable[[Event], Awaitable[None]]


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        self._handlers[event_name].append(handler)

    def unsubscribe(self, event_name: str, handler: EventHandler) -> None:
        if handler in self._handlers.get(event_name, []):
            self._handlers[event_name].remove(handler)

    async def publish(self, event: Event) -> None:
        handlers = self._handlers.get(event.name, [])
        if not handlers:
            logger.debug(
                "No handlers registered for event",
                extra={"extra_fields": {"event_name": event.name}},
            )
            return

        results = await asyncio.gather(
            *(handler(event) for handler in handlers), return_exceptions=True
        )
        for result in results:
            if isinstance(result, Exception):
                logger.error(
                    "Event handler raised an exception",
                    extra={"extra_fields": {"event_name": event.name, "error": str(result)}},
                )


event_bus = EventBus()
