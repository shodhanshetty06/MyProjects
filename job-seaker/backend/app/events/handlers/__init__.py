"""Event handler registrations.

Each module wires its handlers here (or in its own package, imported from
here) so both `app.main` (FastAPI process) and each Celery worker process
can register them all against the shared `event_bus` at startup — the bus
is in-process, so subscriptions made in the API process are invisible to a
worker process and vice versa; every process that might call
`event_bus.publish()` must call `register_all_handlers()` itself (see
`app/workers/celery_app.py`'s `worker_process_init` signal).

Idempotent: calling this twice in the same process would double-subscribe
every handler, so callers must call it exactly once per process (both
call sites already satisfy this — FastAPI's lifespan runs once, and Celery's
`worker_process_init` fires once per worker process).
"""

from app.events.bus import event_bus
from app.events.handlers import job_pipeline


def register_all_handlers() -> None:
    """Register every module's event handlers against the shared event bus.

    Called once from the FastAPI lifespan startup and once per Celery
    worker process startup.
    """
    job_pipeline.register(event_bus)
