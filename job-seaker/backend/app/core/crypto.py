"""Symmetric encryption for secrets stored at rest (e.g. Gmail OAuth tokens).

Uses a dedicated `GMAIL_TOKEN_ENCRYPTION_KEY` rather than deriving a key from
`SECRET_KEY`, so token-encryption key rotation is independent of session/JWT
secret rotation. `cryptography` is already a transitive dependency via
`python-jose[cryptography]` (see `requirements.txt`) — no new package needed.
"""

from __future__ import annotations

from functools import lru_cache

from cryptography.fernet import Fernet

from app.core.config import get_settings
from app.core.exceptions import AppException


class EncryptionKeyMissingError(AppException):
    status_code = 500
    error_code = "encryption_key_missing"
    message = "GMAIL_TOKEN_ENCRYPTION_KEY is not configured."


@lru_cache
def _fernet() -> Fernet:
    settings = get_settings()
    if not settings.GMAIL_TOKEN_ENCRYPTION_KEY:
        raise EncryptionKeyMissingError()
    return Fernet(settings.GMAIL_TOKEN_ENCRYPTION_KEY.encode())


def encrypt_token(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt_token(ciphertext: str) -> str:
    return _fernet().decrypt(ciphertext.encode()).decode()
