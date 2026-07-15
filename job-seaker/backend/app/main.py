"""FastAPI application factory and entrypoint.

Run with:
    uvicorn app.main:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router as api_v1_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db.session import engine
from app.events.handlers import register_all_handlers
from app.integrations.redis_client import close_redis_client, get_redis_client
from app.middleware.exception_middleware import register_exception_handlers
from app.middleware.logging_middleware import LoggingMiddleware
from app.middleware.request_context import RequestContextMiddleware

settings = get_settings()
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Application startup", extra={"extra_fields": {"environment": settings.ENVIRONMENT}})

    # Warm the Redis connection and verify connectivity fails fast if misconfigured.
    redis_client = get_redis_client()
    await redis_client.ping()

    register_all_handlers()

    yield

    logger.info("Application shutdown")
    await close_redis_client()
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.APP_VERSION,
        description=(
            "Enterprise AI-Powered Job Search Platform — API. "
            "Phase 0: foundation only, business logic not yet implemented."
        ),
        docs_url=f"{settings.API_V1_PREFIX}/docs",
        redoc_url=f"{settings.API_V1_PREFIX}/redoc",
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        lifespan=lifespan,
    )

    if settings.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            # Browsers hide all but a small header allowlist from JS by
            # default; the analytics export endpoints need the client to
            # read the server-chosen filename off this header.
            expose_headers=["Content-Disposition"],
        )

    # Order matters: outermost added last runs first. RequestContext must
    # run before LoggingMiddleware so the request_id is set before logging.
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)

    app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)

    return app


app = create_app()
