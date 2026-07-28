"""AWS S3 storage provider — ported from the pre-refactor `S3Storage`
integration. Ready to select via `STORAGE_PROVIDER=s3`; no other
application code changes needed. Also usable against any S3-compatible
endpoint (MinIO/LocalStack) via `S3_ENDPOINT_URL`.
"""

from __future__ import annotations

from functools import lru_cache

import boto3
from botocore.client import BaseClient

from app.core.config import Settings


@lru_cache
def _get_s3_client(
    region: str, access_key: str | None, secret_key: str | None, endpoint_url: str | None
) -> BaseClient:
    return boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint_url or None,
    )


class S3StorageProvider:
    """Object storage over the default S3 bucket."""

    def __init__(self, settings: Settings) -> None:
        self.bucket = settings.S3_BUCKET_NAME
        self.client = _get_s3_client(
            settings.AWS_REGION,
            settings.AWS_ACCESS_KEY_ID,
            settings.AWS_SECRET_ACCESS_KEY,
            settings.S3_ENDPOINT_URL,
        )

    def upload_bytes(self, data: bytes, key: str) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def download_bytes(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def delete_object(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def get_download_url(self, key: str, expires_in: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )


__all__ = ["S3StorageProvider"]
