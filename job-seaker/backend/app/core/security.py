"""Password hashing and JWT/refresh-token helpers.

Lives in `app.core` (parallel to `app.core.crypto`) rather than inside the
`auth` module so `app.api.deps` — which every module's router depends on for
identity — can import it without reaching into a feature module.
"""
from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta, timezone
from hashlib import sha256

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import get_settings
from app.core.exceptions import UnauthorizedError

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ACCESS_TOKEN_TYPE = "access"
REFRESH_COOKIE_NAME = "refresh_token"


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return _pwd_context.verify(password, password_hash)


def create_access_token(user_id: uuid.UUID) -> tuple[str, datetime]:
    """Returns (encoded JWT, expiry). Payload carries only the user id and
    token type — no PII, so it's safe to decode client-side for display."""
    settings = get_settings()
    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {"sub": str(user_id), "type": ACCESS_TOKEN_TYPE, "exp": expires_at}
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return token, expires_at


def decode_access_token(token: str) -> uuid.UUID:
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError as exc:
        raise UnauthorizedError("Invalid or expired access token.") from exc

    if payload.get("type") != ACCESS_TOKEN_TYPE:
        raise UnauthorizedError("Invalid or expired access token.")

    try:
        return uuid.UUID(payload["sub"])
    except (KeyError, ValueError, TypeError) as exc:
        raise UnauthorizedError("Invalid or expired access token.") from exc


def generate_refresh_token() -> tuple[str, str]:
    """Returns (raw token to hand to the client, sha256 hash to persist).

    Only the hash is stored server-side — a leaked DB row can't be replayed
    as a live refresh token, the same reasoning most session-token schemes
    (and `app.core.crypto`'s Fernet-at-rest tokens) already follow here.
    """
    raw = secrets.token_urlsafe(48)
    return raw, hash_refresh_token(raw)


def hash_refresh_token(raw: str) -> str:
    return sha256(raw.encode()).hexdigest()
