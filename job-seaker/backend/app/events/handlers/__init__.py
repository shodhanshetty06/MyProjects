"""Event handler registrations.

Each module wires its handlers here (or in its own package, imported from
here) so `app.main` can register them all against the shared `event_bus`
at startup. Empty in Phase 0.
"""

from app.events.bus import event_bus  # noqa: F401


def register_all_handlers() -> None:
    """Register every module's event handlers against the shared event bus.

    Called once from the FastAPI lifespan startup. No-op until modules
    start publishing/subscribing to real events.
    """
    pass
