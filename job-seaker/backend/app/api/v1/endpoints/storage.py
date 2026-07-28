"""Local-storage signed download route.

Only meaningful when `STORAGE_PROVIDER=local` — `LocalStorageProvider.get_download_url`
(see `app/providers/storage/local_provider.py`) builds URLs against this
route, HMAC-signed with an expiry so no auth header is needed for the
direct-download link (the same trust model a presigned S3 URL has: the
signature itself is the credential).
"""

from __future__ import annotations

import mimetypes

from fastapi import APIRouter, Query, Response

from app.api.deps import SettingsDep, StorageProviderDep
from app.core.exceptions import UnauthorizedError
from app.providers.storage.local_provider import verify_local_storage_url

router = APIRouter(prefix="/storage", tags=["Storage"])


@router.get("/local/{key:path}")
async def download_local_object(
    key: str,
    settings: SettingsDep,
    storage: StorageProviderDep,
    expires_at: int = Query(...),
    signature: str = Query(...),
) -> Response:
    if not verify_local_storage_url(
        key=key,
        expires_at=expires_at,
        signature=signature,
        secret=settings.local_storage_signing_secret,
    ):
        raise UnauthorizedError("Invalid or expired download link.")

    data = storage.download_bytes(key)
    content_type, _ = mimetypes.guess_type(key)
    return Response(content=data, media_type=content_type or "application/octet-stream")
