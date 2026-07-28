"""Contract every file-storage provider (local filesystem, S3, and future
R2/Azure/GCS backends) must satisfy so callers never import boto3 or touch
the filesystem directly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class StorageProvider(Protocol):
    """A binary object store keyed by a path-like string."""

    bucket: str

    def upload_bytes(self, data: bytes, key: str) -> None: ...

    def download_bytes(self, key: str) -> bytes: ...

    def delete_object(self, key: str) -> None: ...

    def get_download_url(self, key: str, expires_in: int = 3600) -> str:
        """A time-limited URL a client can use to download `key` directly."""
        ...
