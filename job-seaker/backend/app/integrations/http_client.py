"""Shared outbound HTTP client factory for external API calls (ATS adapters).

Centralizes timeout and User-Agent configuration so individual adapters
don't each construct their own `httpx.AsyncClient` from scratch — mirrors
the `get_s3_client()` factory pattern in `app/integrations/s3_client.py`.

Deliberately *not* cached/singleton: Celery tasks run each pipeline step via
a fresh `asyncio.run(...)` call (see `tasks.py`), which means a new event
loop per task. An `httpx.AsyncClient`'s connection pool is bound to the loop
it was created on, so a cached client would break on the second task. Build
one per task with `async with build_http_client() as client:` instead.
"""

from __future__ import annotations

import httpx

from app.core.config import get_settings


def build_http_client() -> httpx.AsyncClient:
    settings = get_settings()
    return httpx.AsyncClient(
        timeout=settings.JOB_INGESTION_HTTP_TIMEOUT_SECONDS,
        headers={"User-Agent": settings.JOB_INGESTION_USER_AGENT},
        follow_redirects=True,
    )
