"""Lightweight dependency-injection container.

FastAPI's `Depends` already provides request-scoped DI for HTTP handlers
(see `app/api/deps.py`). This container exists for the cases FastAPI's
graph doesn't cover: Celery tasks, event handlers, and scripts that need the
same singletons (settings, redis, storage/email providers) without a
request in flight. Moved here (from the former `app/core/di.py`) as part of
the provider-pattern refactor so DI wiring lives in one dedicated package.

Usage:
    container = get_container()
    ai_service = container.resolve("ai_service")
"""
from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Container:
    """A minimal service locator supporting singleton and factory bindings."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Any]] = {}
        self._singletons: dict[str, Any] = {}

    def register_factory(self, key: str, factory: Callable[[], Any]) -> None:
        """Register a callable that produces a new instance on every resolve()."""
        self._factories[key] = factory

    def register_singleton(self, key: str, factory: Callable[[], Any]) -> None:
        """Register a callable whose result is cached after the first resolve()."""
        self._factories[key] = factory
        self._singletons.pop(key, None)

    def resolve(self, key: str) -> Any:
        if key in self._singletons:
            return self._singletons[key]

        if key not in self._factories:
            raise KeyError(f"No binding registered for '{key}'")

        instance = self._factories[key]()

        # Anything registered via register_singleton is cached; plain
        # register_factory bindings are recreated every call.
        if key not in self._singletons and key in _SINGLETON_KEYS:
            self._singletons[key] = instance

        return instance


_SINGLETON_KEYS: set[str] = set()
_container: Container | None = None


def get_container() -> Container:
    global _container
    if _container is None:
        _container = _build_container()
    return _container


def _build_container() -> Container:
    """Wire up the default application bindings.

    Kept lazy (imports inside the function) to avoid import cycles between
    core, db, and integrations at module load time.

    `ai_service`/`embedding_service` are plain factories (a fresh instance
    per resolve) rather than singletons — each holds no expensive state of
    its own (the underlying provider SDK clients/models they wrap are
    already cached at the provider layer via `@lru_cache`), and a fresh
    instance sidesteps any risk of two concurrent Celery-task coroutines
    sharing retry state. `storage_provider`/`email_provider` are singletons,
    matching the previous `s3_client` binding's caching behavior.
    """
    from app.core.config import get_settings
    from app.dependency_injection.providers import (
        build_ai_service,
        build_email_provider_instance,
        build_embedding_service,
        build_storage_provider_instance,
    )
    from app.integrations.redis_client import get_redis_client

    container = Container()

    _SINGLETON_KEYS.update({"settings", "redis_client", "storage_provider", "email_provider"})

    container.register_singleton("settings", get_settings)
    container.register_singleton("redis_client", get_redis_client)
    container.register_singleton(
        "storage_provider", lambda: build_storage_provider_instance(get_settings())
    )
    container.register_singleton(
        "email_provider", lambda: build_email_provider_instance(get_settings())
    )
    container.register_factory("ai_service", lambda: build_ai_service(get_settings()))
    container.register_factory(
        "embedding_service",
        lambda: build_embedding_service(get_settings(), get_redis_client()),
    )

    return container
