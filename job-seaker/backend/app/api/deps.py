"""Common FastAPI dependencies shared across every module's router.

Exposes the pieces every module router needs: DB session, Redis client, S3
storage, event bus, and the authenticated caller's identity.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.exceptions import UnauthorizedError
from app.core.security import decode_access_token
from app.db.session import get_db
from app.events.bus import EventBus, event_bus
from app.integrations.cache import CacheClient
from app.integrations.redis_client import get_redis_client
from app.integrations.s3_client import S3Storage, get_s3_client

SettingsDep = Annotated[Settings, Depends(get_settings)]
DBSessionDep = Annotated[AsyncSession, Depends(get_db)]
RedisDep = Annotated[Redis, Depends(get_redis_client)]


def get_cache_client(redis: RedisDep) -> CacheClient:
    return CacheClient(redis)


CacheDep = Annotated[CacheClient, Depends(get_cache_client)]


def get_s3_storage() -> S3Storage:
    return S3Storage(client=get_s3_client())


S3StorageDep = Annotated[S3Storage, Depends(get_s3_storage)]


def get_event_bus() -> EventBus:
    return event_bus


EventBusDep = Annotated[EventBus, Depends(get_event_bus)]


_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user_id(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)] = None,
) -> uuid.UUID:
    """Resolve the requesting user's ID from a verified JWT access token.

    Every module router depends on this (directly or via `CurrentUserIdDep`)
    for identity, so this is the one place a bearer token is checked.
    """
    if credentials is None or not credentials.credentials:
        raise UnauthorizedError("Authentication required.")
    return decode_access_token(credentials.credentials)


CurrentUserIdDep = Annotated[uuid.UUID, Depends(get_current_user_id)]
