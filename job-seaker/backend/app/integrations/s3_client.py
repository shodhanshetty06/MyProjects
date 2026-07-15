"""AWS S3 client management and thin storage helpers.

Wraps boto3 so callers depend on this module instead of importing boto3
directly — keeps the AWS SDK an implementation detail that can be swapped
(e.g. for MinIO/LocalStack in tests) via `S3_ENDPOINT_URL`.
"""
from __future__ import annotations

from functools import lru_cache

import boto3
from botocore.client import BaseClient

from app.core.config import get_settings


@lru_cache
def get_s3_client() -> BaseClient:
    settings = get_settings()
    return boto3.client(
        "s3",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        endpoint_url=settings.S3_ENDPOINT_URL or None,
    )


class S3Storage:
    """Thin convenience wrapper around the S3 client for the default bucket."""

    def __init__(self, client: BaseClient | None = None, bucket: str | None = None) -> None:
        settings = get_settings()
        self.client = client or get_s3_client()
        self.bucket = bucket or settings.S3_BUCKET_NAME

    def generate_presigned_upload_url(self, key: str, expires_in: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    def generate_presigned_download_url(self, key: str, expires_in: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    def upload_file(self, file_path: str, key: str) -> None:
        self.client.upload_file(file_path, self.bucket, key)

    def upload_bytes(self, data: bytes, key: str) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def delete_object(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def download_bytes(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()
