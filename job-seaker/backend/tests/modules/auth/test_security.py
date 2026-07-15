"""Unit tests for `app.core.security` — password hashing and JWT/refresh
token helpers. Pure functions, no Postgres/Redis required."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from jose import jwt

from app.core.config import get_settings
from app.core.exceptions import UnauthorizedError
from app.core.security import (
    create_access_token,
    decode_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)


def test_hash_password_does_not_return_plaintext() -> None:
    hashed = hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"


def test_verify_password_accepts_correct_and_rejects_wrong() -> None:
    hashed = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", hashed) is True
    assert verify_password("wrong password", hashed) is False


def test_create_and_decode_access_token_round_trips() -> None:
    user_id = uuid.uuid4()
    token, expires_at = create_access_token(user_id)
    assert decode_access_token(token) == user_id
    assert expires_at > datetime.now(timezone.utc)


def test_decode_access_token_rejects_garbage() -> None:
    with pytest.raises(UnauthorizedError):
        decode_access_token("not-a-jwt")


def test_decode_access_token_rejects_expired_token() -> None:
    settings = get_settings()
    payload = {
        "sub": str(uuid.uuid4()),
        "type": "access",
        "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
    }
    expired = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    with pytest.raises(UnauthorizedError):
        decode_access_token(expired)


def test_decode_access_token_rejects_wrong_token_type() -> None:
    settings = get_settings()
    payload = {
        "sub": str(uuid.uuid4()),
        "type": "refresh",
        "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
    }
    wrong_type = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    with pytest.raises(UnauthorizedError):
        decode_access_token(wrong_type)


def test_decode_access_token_rejects_token_signed_with_a_different_key() -> None:
    settings = get_settings()
    payload = {
        "sub": str(uuid.uuid4()),
        "type": "access",
        "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
    }
    forged = jwt.encode(payload, "a-different-secret", algorithm=settings.ALGORITHM)
    with pytest.raises(UnauthorizedError):
        decode_access_token(forged)


def test_generate_refresh_token_hash_matches_hash_refresh_token() -> None:
    raw, digest = generate_refresh_token()
    assert digest == hash_refresh_token(raw)


def test_generate_refresh_token_is_unique_per_call() -> None:
    raw_one, _ = generate_refresh_token()
    raw_two, _ = generate_refresh_token()
    assert raw_one != raw_two
