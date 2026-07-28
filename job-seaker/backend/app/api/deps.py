"""Common FastAPI dependencies shared across every module's router.

Exposes the pieces every module router needs: DB session, Redis client,
storage, email, AI/embedding services, event bus, and the authenticated
caller's identity.
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
from app.dependency_injection.providers import (
    build_ai_service,
    build_email_provider_instance,
    build_embedding_service,
    build_storage_provider_instance,
)
from app.events.bus import EventBus, event_bus
from app.integrations.cache import CacheClient
from app.integrations.redis_client import get_redis_client
from app.interfaces.email_provider import EmailProvider
from app.interfaces.storage_provider import StorageProvider
from app.services.ai.service import AIService
from app.services.embedding.service import EmbeddingService

SettingsDep = Annotated[Settings, Depends(get_settings)]
DBSessionDep = Annotated[AsyncSession, Depends(get_db)]
RedisDep = Annotated[Redis, Depends(get_redis_client)]


def get_cache_client(redis: RedisDep) -> CacheClient:
    return CacheClient(redis)


CacheDep = Annotated[CacheClient, Depends(get_cache_client)]


def get_ai_service(settings: SettingsDep) -> AIService:
    return build_ai_service(settings)


AIServiceDep = Annotated[AIService, Depends(get_ai_service)]


def get_embedding_service(settings: SettingsDep, redis: RedisDep) -> EmbeddingService:
    return build_embedding_service(settings, redis)


EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]


def get_storage_provider(settings: SettingsDep) -> StorageProvider:
    return build_storage_provider_instance(settings)


StorageProviderDep = Annotated[StorageProvider, Depends(get_storage_provider)]


def get_email_provider(settings: SettingsDep) -> EmailProvider:
    return build_email_provider_instance(settings)


EmailProviderDep = Annotated[EmailProvider, Depends(get_email_provider)]


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


def get_optional_current_user_id(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)] = None,
) -> uuid.UUID | None:
    """Same bearer-token decode as `get_current_user_id`, but for endpoints
    that must also serve anonymous callers (e.g. `GET /search/jobs/{id}` —
    consistent with plain search being public). Returns `None` instead of
    raising 401 when no/invalid credentials are present; a *malformed* token
    still isn't silently treated as "no token" beyond that — it's simply
    resolved to `None` the same as absent credentials, so a bad token never
    surfaces a 401 on an otherwise-public endpoint.
    """
    if credentials is None or not credentials.credentials:
        return None
    try:
        return decode_access_token(credentials.credentials)
    except UnauthorizedError:
        return None


OptionalCurrentUserIdDep = Annotated[uuid.UUID | None, Depends(get_optional_current_user_id)]
