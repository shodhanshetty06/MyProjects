"""Picks the concrete `StorageProvider` implementation from
`settings.STORAGE_PROVIDER`.

`r2`/`azure`/`gcs` are recognized as valid future values (per the spec's
Provider Pattern goal) but raise `NotImplementedError` until a concrete
`providers/storage/*` class is added for them — no config schema for those
was specified, unlike `local`/`s3`.
"""

from __future__ import annotations

from app.core.config import Settings
from app.interfaces.storage_provider import StorageProvider

_FUTURE_PROVIDERS = {"r2", "azure", "gcs"}


def build_storage_provider(settings: Settings) -> StorageProvider:
    provider = settings.STORAGE_PROVIDER.lower()

    if provider == "local":
        from app.providers.storage.local_provider import LocalStorageProvider

        return LocalStorageProvider(settings)
    if provider == "s3":
        from app.providers.storage.s3_provider import S3StorageProvider

        return S3StorageProvider(settings)
    if provider in _FUTURE_PROVIDERS:
        raise NotImplementedError(
            f"STORAGE_PROVIDER '{provider}' is not implemented yet. "
            "Add a providers/storage/{provider}_provider.py conforming to StorageProvider."
        )

    raise ValueError(f"Unknown STORAGE_PROVIDER '{settings.STORAGE_PROVIDER}'. Expected: local, s3.")


__all__ = ["build_storage_provider"]
