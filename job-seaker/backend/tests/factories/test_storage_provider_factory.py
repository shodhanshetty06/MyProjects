"""Provider-switching tests for `build_storage_provider`."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.factories.storage_provider_factory import build_storage_provider
from app.providers.storage.local_provider import LocalStorageProvider
from app.providers.storage.s3_provider import S3StorageProvider


def test_build_storage_provider_selects_local(tmp_path) -> None:
    settings = Settings(STORAGE_PROVIDER="local", LOCAL_STORAGE_PATH=str(tmp_path))

    provider = build_storage_provider(settings)

    assert isinstance(provider, LocalStorageProvider)


def test_build_storage_provider_selects_s3() -> None:
    settings = Settings(STORAGE_PROVIDER="s3")

    provider = build_storage_provider(settings)

    assert isinstance(provider, S3StorageProvider)


@pytest.mark.parametrize("provider_name", ["r2", "azure", "gcs"])
def test_build_storage_provider_future_providers_raise_not_implemented(provider_name) -> None:
    settings = Settings(STORAGE_PROVIDER=provider_name)

    with pytest.raises(NotImplementedError):
        build_storage_provider(settings)


def test_build_storage_provider_rejects_unknown_provider() -> None:
    settings = Settings(STORAGE_PROVIDER="not-a-real-provider")

    with pytest.raises(ValueError, match="Unknown STORAGE_PROVIDER"):
        build_storage_provider(settings)
