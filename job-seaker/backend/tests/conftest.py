"""Shared pytest fixtures."""
from __future__ import annotations

import os
import re
from typing import AsyncIterator

# Point the app at a dedicated test database *before* anything imports
# app.db.session (which builds its engine from settings at import time).
# Without this, tests write directly into the dev/prod database — see the
# 2026-07 incident where pytest fixtures (hex-suffixed job/company names)
# permanently polluted search results in the running app.
_default_url = "postgresql+asyncpg://jobplatform:jobplatform@postgres:5432/jobplatform"
_app_url = os.environ.get("DATABASE_URL", _default_url)
os.environ["DATABASE_URL"] = re.sub(r"/([^/?]+)(\?.*)?$", r"/jobplatform_test\2", _app_url)

import asyncio

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient

from app.db.base_class import Base
from app.main import app


@pytest.fixture(scope="session", autouse=True)
def _reset_test_database() -> None:
    """Truncate every app table before the suite runs.

    Keeps the dedicated jobplatform_test database (see DATABASE_URL override
    above) from accumulating fixture rows across repeated local test runs.
    Uses its own short-lived event loop and connection via asyncpg directly
    — deliberately independent of app.db.session's engine/pool, since that
    engine is a module-level singleton and pytest-asyncio hands each test
    function its own event loop, so sharing pooled asyncpg connections
    across tests raises "Event loop is closed" / protocol errors.
    """
    table_names = [t.name for t in Base.metadata.sorted_tables]
    if not table_names:
        return

    async def _truncate() -> None:
        dsn = os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://")
        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute(
                f"TRUNCATE TABLE {', '.join(table_names)} RESTART IDENTITY CASCADE"
            )
        finally:
            await conn.close()

    asyncio.run(_truncate())


@pytest.fixture(autouse=True)
def _dispose_stale_event_loop_bound_resources() -> None:
    """Both the SQLAlchemy async engine's pooled connections and the
    `@lru_cache`d Redis client are bound to whichever asyncio event loop was
    running when they were opened. pytest-asyncio's default fixture scope
    hands each test function its own event loop, so a resource opened by
    test N (under test N's loop) reused by test N+1 (a different loop)
    raises "Future attached to a different loop" / "Event loop is closed" —
    the identical symptom `app/workers/celery_app.py::_reset_async_resources`
    already documents and works around between Celery tasks. Same fix here,
    run before every test: `dispose(close=False)` drops the pool's
    references to already-stale connections (no attempt to gracefully close
    a connection whose loop is gone), and clearing the Redis client cache
    forces a fresh client on the next test's own loop.
    """
    from app.db.session import engine
    from app.integrations.redis_client import get_redis_client

    engine.sync_engine.dispose(close=False)
    get_redis_client.cache_clear()


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
