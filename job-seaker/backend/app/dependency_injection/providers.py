"""Single source of truth for building `AIService`/`EmbeddingService`/
`StorageProvider`/`EmailProvider` instances. Both `app/api/deps.py` (FastAPI
`Depends`, HTTP request context) and `app/dependency_injection/container.py`
(Celery tasks, no request in flight) call these — neither constructs a
provider directly, so there's exactly one place that wires a `Settings`
value to a concrete provider class.
"""

from __future__ import annotations

from redis.asyncio import Redis

from app.core.config import Settings
from app.factories.ai_provider_factory import build_ai_provider
from app.factories.email_provider_factory import build_email_provider
from app.factories.embedding_provider_factory import build_embedding_provider
from app.factories.storage_provider_factory import build_storage_provider
from app.integrations.cache import CacheClient
from app.interfaces.email_provider import EmailProvider
from app.interfaces.storage_provider import StorageProvider
from app.services.ai.service import AIService
from app.services.embedding.service import EmbeddingService


def build_ai_service(settings: Settings) -> AIService:
    return AIService(build_ai_provider(settings), settings)


def build_embedding_service(settings: Settings, redis_client: Redis | None = None) -> EmbeddingService:
    cache = CacheClient(redis_client) if redis_client is not None else None
    return EmbeddingService(build_embedding_provider(settings), settings, cache=cache)


def build_storage_provider_instance(settings: Settings) -> StorageProvider:
    return build_storage_provider(settings)


def build_email_provider_instance(settings: Settings) -> EmailProvider:
    return build_email_provider(settings)


__all__ = [
    "build_ai_service",
    "build_embedding_service",
    "build_storage_provider_instance",
    "build_email_provider_instance",
]
