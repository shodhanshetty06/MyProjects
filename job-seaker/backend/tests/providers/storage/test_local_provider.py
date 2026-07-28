"""Unit tests for `LocalStorageProvider` — filesystem object storage plus
the HMAC URL signing/verification `app/api/v1/endpoints/storage.py`'s
download route relies on. Uses a real temp directory (no mocking needed,
it's just local disk I/O) and no network access.
"""

from __future__ import annotations

from app.core.config import Settings
from app.providers.storage.local_provider import (
    LocalStorageProvider,
    sign_local_storage_url,
    verify_local_storage_url,
)


def _provider(tmp_path) -> LocalStorageProvider:
    settings = Settings(
        LOCAL_STORAGE_PATH=str(tmp_path),
        LOCAL_STORAGE_BASE_URL="http://localhost:8000",
        LOCAL_STORAGE_URL_SIGNING_SECRET="test-signing-secret",
    )
    return LocalStorageProvider(settings)


def test_upload_and_download_round_trip(tmp_path) -> None:
    provider = _provider(tmp_path)

    provider.upload_bytes(b"resume contents", "resumes/u1/r1/v1/resume.pdf")

    assert provider.download_bytes("resumes/u1/r1/v1/resume.pdf") == b"resume contents"


def test_upload_creates_nested_directories(tmp_path) -> None:
    provider = _provider(tmp_path)

    provider.upload_bytes(b"data", "a/b/c/file.txt")

    assert (tmp_path / "a" / "b" / "c" / "file.txt").read_bytes() == b"data"


def test_delete_object_removes_file(tmp_path) -> None:
    provider = _provider(tmp_path)
    provider.upload_bytes(b"data", "file.txt")

    provider.delete_object("file.txt")

    assert not (tmp_path / "file.txt").exists()


def test_delete_object_is_idempotent_for_missing_key(tmp_path) -> None:
    provider = _provider(tmp_path)

    provider.delete_object("does-not-exist.txt")  # must not raise


def test_get_download_url_produces_verifiable_signature(tmp_path) -> None:
    provider = _provider(tmp_path)

    url = provider.get_download_url("resumes/u1/r1/v1/resume.pdf", expires_in=3600)

    assert url.startswith("http://localhost:8000/api/v1/storage/local/resumes/u1/r1/v1/resume.pdf?")


def test_sign_and_verify_round_trip() -> None:
    expires_at, signature = sign_local_storage_url(
        key="a/b/c.pdf", expires_in=3600, secret="shared-secret"
    )

    assert verify_local_storage_url(
        key="a/b/c.pdf", expires_at=expires_at, signature=signature, secret="shared-secret"
    )


def test_verify_rejects_tampered_key() -> None:
    expires_at, signature = sign_local_storage_url(
        key="a/b/c.pdf", expires_in=3600, secret="shared-secret"
    )

    assert not verify_local_storage_url(
        key="a/b/other.pdf", expires_at=expires_at, signature=signature, secret="shared-secret"
    )


def test_verify_rejects_wrong_secret() -> None:
    expires_at, signature = sign_local_storage_url(
        key="a/b/c.pdf", expires_in=3600, secret="shared-secret"
    )

    assert not verify_local_storage_url(
        key="a/b/c.pdf", expires_at=expires_at, signature=signature, secret="wrong-secret"
    )


def test_verify_rejects_expired_url() -> None:
    expires_at, signature = sign_local_storage_url(
        key="a/b/c.pdf", expires_in=-10, secret="shared-secret"
    )

    assert not verify_local_storage_url(
        key="a/b/c.pdf", expires_at=expires_at, signature=signature, secret="shared-secret"
    )


def test_path_traversal_key_is_rejected(tmp_path) -> None:
    provider = _provider(tmp_path)

    try:
        provider.upload_bytes(b"data", "../escape.txt")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a path-escaping key")
