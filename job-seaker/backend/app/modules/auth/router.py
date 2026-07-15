"""Authentication module — API routes (v1).

`POST /login` and `POST /refresh` set the rotating refresh token as an
httpOnly cookie scoped to this router's own path prefix — the browser only
ever attaches it to `/auth/refresh` and `/auth/logout`, never to the rest of
the API, so an XSS bug elsewhere in the SPA can't read or exfiltrate it.
The short-lived access token is returned in the response body instead, for
the frontend to hold in memory and send as `Authorization: Bearer`.
"""
from __future__ import annotations

from fastapi import APIRouter, Cookie, Depends, Response, status

from app.api.deps import CurrentUserIdDep, DBSessionDep
from app.core.config import get_settings
from app.core.exceptions import UnauthorizedError
from app.core.security import REFRESH_COOKIE_NAME
from app.modules.auth.repository import AuthRepository
from app.modules.auth.schemas import AccessTokenResponse, LoginRequest, RegisterRequest, UserRead
from app.modules.auth.service import AuthService, IssuedTokens
from app.schemas.common import BaseResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])

_settings = get_settings()
_REFRESH_COOKIE_PATH = f"{_settings.API_V1_PREFIX}/auth"


def get_auth_service(session: DBSessionDep) -> AuthService:
    return AuthService(AuthRepository(session))


ServiceDep = Depends(get_auth_service)
RefreshCookieDep = Cookie(default=None, alias=REFRESH_COOKIE_NAME)


def _set_refresh_cookie(response: Response, raw_refresh_token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=raw_refresh_token,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        path=_REFRESH_COOKIE_PATH,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite="strict",
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(key=REFRESH_COOKIE_NAME, path=_REFRESH_COOKIE_PATH)


def _token_response(tokens: IssuedTokens, message: str) -> BaseResponse[AccessTokenResponse]:
    return BaseResponse(
        message=message,
        data=AccessTokenResponse(
            access_token=tokens.access_token,
            expires_at=tokens.access_token_expires_at,
            user=UserRead.model_validate(tokens.user),
        ),
    )


@router.post("/register", response_model=BaseResponse[AccessTokenResponse], status_code=status.HTTP_201_CREATED)
async def register(
    payload: RegisterRequest,
    response: Response,
    service: AuthService = ServiceDep,
) -> BaseResponse[AccessTokenResponse]:
    user = await service.register(email=payload.email, password=payload.password)
    tokens = await service.issue_tokens(user)
    _set_refresh_cookie(response, tokens.refresh_token)
    return _token_response(tokens, "Account created.")


@router.post("/login", response_model=BaseResponse[AccessTokenResponse])
async def login(
    payload: LoginRequest,
    response: Response,
    service: AuthService = ServiceDep,
) -> BaseResponse[AccessTokenResponse]:
    user = await service.authenticate(email=payload.email, password=payload.password)
    tokens = await service.issue_tokens(user)
    _set_refresh_cookie(response, tokens.refresh_token)
    return _token_response(tokens, "Logged in.")


@router.post("/refresh", response_model=BaseResponse[AccessTokenResponse])
async def refresh(
    response: Response,
    refresh_token: str | None = RefreshCookieDep,
    service: AuthService = ServiceDep,
) -> BaseResponse[AccessTokenResponse]:
    if not refresh_token:
        raise UnauthorizedError("No refresh token provided.")
    tokens = await service.refresh(refresh_token)
    _set_refresh_cookie(response, tokens.refresh_token)
    return _token_response(tokens, "Token refreshed.")


@router.post("/logout", response_model=BaseResponse[None])
async def logout(
    response: Response,
    refresh_token: str | None = RefreshCookieDep,
    service: AuthService = ServiceDep,
) -> BaseResponse[None]:
    await service.logout(refresh_token)
    _clear_refresh_cookie(response)
    return BaseResponse(message="Logged out.")


@router.get("/me", response_model=BaseResponse[UserRead])
async def get_me(
    user_id: CurrentUserIdDep,
    service: AuthService = ServiceDep,
) -> BaseResponse[UserRead]:
    user = await service.get_user(user_id)
    return BaseResponse(data=UserRead.model_validate(user))
