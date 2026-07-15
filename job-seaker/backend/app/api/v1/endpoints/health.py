"""Health check endpoints.

`/health` — liveness: process is up.
`/health/ready` — readiness: process + its critical dependencies
(Postgres, Redis) are reachable. Used by orchestrators (Docker healthcheck,
k8s readiness probe, load balancer) to gate traffic.
"""
from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import text

from app.api.deps import DBSessionDep, RedisDep
from app.core.config import get_settings
from app.core.constants import HealthStatus
from app.schemas.common import BaseResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=BaseResponse)
async def liveness() -> BaseResponse:
    settings = get_settings()
    return BaseResponse(
        message="Service is alive.",
        data={
            "status": HealthStatus.OK,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
        },
    )


@router.get("/ready", response_model=BaseResponse)
async def readiness(db: DBSessionDep, redis: RedisDep) -> BaseResponse:
    dependencies: dict[str, str] = {}
    overall = HealthStatus.OK

    try:
        await db.execute(text("SELECT 1"))
        dependencies["postgres"] = HealthStatus.OK
    except Exception:
        dependencies["postgres"] = HealthStatus.DOWN
        overall = HealthStatus.DEGRADED

    try:
        await redis.ping()
        dependencies["redis"] = HealthStatus.OK
    except Exception:
        dependencies["redis"] = HealthStatus.DOWN
        overall = HealthStatus.DEGRADED

    return BaseResponse(
        message="Readiness check complete.",
        data={"status": overall, "dependencies": dependencies},
    )
