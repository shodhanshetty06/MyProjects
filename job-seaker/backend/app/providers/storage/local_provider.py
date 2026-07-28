"""Local filesystem storage provider — the primary/default `StorageProvider`
for development. Stores files under `LOCAL_STORAGE_PATH`; `get_download_url`
signs `key` + an expiry with HMAC-SHA256 (since local disk has no native
presigned-URL concept) and points at `GET /api/v1/storage/local/{key}`
(`app/api/v1/storage_router.py`), which verifies the signature before
streaming the file back.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from pathlib import Path
from urllib.parse import quote, urlencode

from app.core.config import Settings
from app.core.exceptions import NotFoundError


def sign_local_storage_url(*, key: str, expires_in: int, secret: str) -> tuple[int, str]:
    expires_at = int(time.time()) + expires_in
    signature = _signature(key=key, expires_at=expires_at, secret=secret)
    return expires_at, signature


def verify_local_storage_url(*, key: str, expires_at: int, signature: str, secret: str) -> bool:
    if time.time() > expires_at:
        return False
    expected = _signature(key=key, expires_at=expires_at, secret=secret)
    return hmac.compare_digest(expected, signature)


def _signature(*, key: str, expires_at: int, secret: str) -> str:
    payload = f"{key}:{expires_at}".encode()
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


class LocalStorageProvider:
    """Object storage on the local filesystem."""

    def __init__(self, settings: Settings) -> None:
        self.bucket = "local"
        self._root = Path(settings.LOCAL_STORAGE_PATH)
        self._base_url = settings.LOCAL_STORAGE_BASE_URL.rstrip("/")
        self._signing_secret = settings.local_storage_signing_secret

    def _path_for(self, key: str) -> Path:
        path = (self._root / key).resolve()
        if self._root.resolve() not in path.parents and path != self._root.resolve():
            raise ValueError(f"Storage key escapes the storage root: {key!r}")
        return path

    def upload_bytes(self, data: bytes, key: str) -> None:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def download_bytes(self, key: str) -> bytes:
        path = self._path_for(key)
        if not path.is_file():
            raise NotFoundError(f"Storage object not found: {key}")
        return path.read_bytes()

    def delete_object(self, key: str) -> None:
        path = self._path_for(key)
        path.unlink(missing_ok=True)

    def get_download_url(self, key: str, expires_in: int = 3600) -> str:
        expires_at, signature = sign_local_storage_url(
            key=key, expires_in=expires_in, secret=self._signing_secret
        )
        query = urlencode({"expires_at": expires_at, "signature": signature})
        # `safe="/"` keeps path separators literal — the download route
        # matches `{key:path}`, which expects real slashes, not %2F.
        return f"{self._base_url}/api/v1/storage/local/{quote(key, safe='/')}?{query}"


__all__ = ["LocalStorageProvider", "sign_local_storage_url", "verify_local_storage_url"]
