"""Email Tracking module — API routes (v1).

Gmail OAuth connect/disconnect, the manual-review queue for emails the
classifier couldn't confidently act on, and resolution of those reviews.
Every authenticated endpoint is scoped to the caller via `CurrentUserIdDep`
(trusted `X-User-Id` header — see `app/api/deps.py`), except the OAuth
callback which Google itself hits with a bare, header-less redirect.
"""

from __future__ import annotations

import math
import uuid

from fastapi import APIRouter, Depends

from app.api.deps import AIServiceDep, CacheDep, CurrentUserIdDep, DBSessionDep
from app.modules.email_tracking.repository import GmailAccountRepository, TrackedEmailRepository
from app.modules.email_tracking.schemas import (
    GmailAccountRead,
    GmailConnectResponse,
    TrackedEmailDetail,
    TrackedEmailFilters,
    TrackedEmailRead,
    TrackedEmailResolveRequest,
)
from app.modules.email_tracking.service import EmailTrackingService
from app.modules.notification.repository import (
    NotificationPreferenceRepository,
    NotificationRepository,
)
from app.modules.notification.service import NotificationService
from app.modules.tracking.repository import (
    ApplicationNoteRepository,
    ApplicationRepository,
    ApplicationStatusHistoryRepository,
)
from app.modules.tracking.service import TrackingService
from app.schemas.common import BaseResponse, PaginatedResponse, PaginationMeta, PaginationParams

router = APIRouter(prefix="/email-tracking", tags=["Email Tracking"])


def get_email_tracking_service(
    session: DBSessionDep, cache: CacheDep, ai_service: AIServiceDep
) -> EmailTrackingService:
    notification_service = NotificationService(
        repository=NotificationRepository(session),
        preference_repository=NotificationPreferenceRepository(session),
    )
    tracking_service = TrackingService(
        application_repository=ApplicationRepository(session),
        status_history_repository=ApplicationStatusHistoryRepository(session),
        note_repository=ApplicationNoteRepository(session),
        notification_service=notification_service,
    )
    return EmailTrackingService(
        gmail_account_repository=GmailAccountRepository(session),
        tracked_email_repository=TrackedEmailRepository(session),
        application_repository=ApplicationRepository(session),
        tracking_service=tracking_service,
        notification_service=notification_service,
        cache=cache,
        ai_service=ai_service,
    )


ServiceDep = Depends(get_email_tracking_service)


@router.get("/gmail/connect", response_model=BaseResponse[GmailConnectResponse])
async def connect_gmail(user_id: CurrentUserIdDep, service: EmailTrackingService = ServiceDep):
    authorization_url = await service.start_connect(user_id=user_id)
    return BaseResponse(data=GmailConnectResponse(authorization_url=authorization_url))


@router.get("/gmail/callback", response_model=BaseResponse[GmailAccountRead])
async def gmail_callback(code: str, state: str, service: EmailTrackingService = ServiceDep):
    account = await service.complete_connect(code=code, state=state)
    return BaseResponse(data=GmailAccountRead.model_validate(account))


@router.get("/accounts", response_model=BaseResponse[list[GmailAccountRead]])
async def list_accounts(user_id: CurrentUserIdDep, service: EmailTrackingService = ServiceDep):
    accounts = await service.list_accounts(user_id=user_id)
    return BaseResponse(data=[GmailAccountRead.model_validate(account) for account in accounts])


@router.delete("/accounts/{account_id}", response_model=BaseResponse[None])
async def disconnect_account(
    account_id: uuid.UUID, user_id: CurrentUserIdDep, service: EmailTrackingService = ServiceDep
):
    await service.disconnect(user_id=user_id, account_id=account_id)
    return BaseResponse(message="Gmail account disconnected.")


@router.get("/emails", response_model=PaginatedResponse[TrackedEmailRead])
async def list_tracked_emails(
    user_id: CurrentUserIdDep,
    filters: TrackedEmailFilters = Depends(),
    pagination: PaginationParams = Depends(),
    service: EmailTrackingService = ServiceDep,
):
    emails, total = await service.list_tracked_emails(
        user_id=user_id, filters=filters, offset=pagination.offset, limit=pagination.page_size
    )
    return PaginatedResponse(
        data=[TrackedEmailRead.model_validate(email) for email in emails],
        pagination=PaginationMeta(
            page=pagination.page,
            page_size=pagination.page_size,
            total_items=total,
            total_pages=math.ceil(total / pagination.page_size) if total else 0,
        ),
    )


@router.get("/emails/{email_id}", response_model=BaseResponse[TrackedEmailDetail])
async def get_tracked_email(
    email_id: uuid.UUID, user_id: CurrentUserIdDep, service: EmailTrackingService = ServiceDep
):
    email = await service.get_tracked_email(user_id=user_id, email_id=email_id)
    return BaseResponse(data=TrackedEmailDetail.model_validate(email))


@router.post("/emails/{email_id}/resolve", response_model=BaseResponse[TrackedEmailRead])
async def resolve_tracked_email(
    email_id: uuid.UUID,
    payload: TrackedEmailResolveRequest,
    user_id: CurrentUserIdDep,
    service: EmailTrackingService = ServiceDep,
):
    email = await service.resolve_tracked_email(user_id=user_id, email_id=email_id, payload=payload)
    return BaseResponse(data=TrackedEmailRead.model_validate(email))
