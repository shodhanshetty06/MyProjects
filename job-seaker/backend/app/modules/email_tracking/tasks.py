"""Email Tracking module — Celery tasks.

Three tasks forming the Gmail sync pipeline (see `app/workers/celery_app.py`
`task_routes`/`beat_schedule` and the `worker-email-tracking` service in
`docker-compose.yml`):

    sweep_gmail_accounts_task     -> "email_tracking" queue (Celery Beat, periodic)
    poll_gmail_account_task       -> "email_tracking" queue
    fetch_and_classify_email_task -> "email_tracking" queue

Thin sync wrappers around `asyncio.run(...)`, using `AsyncSessionLocal`
directly — same shape as every other module's `tasks.py`. Retry: only
`TransientGmailError` (network/timeout/5xx/429) is handed to Celery's own
`autoretry_for` + `retry_backoff`. `GmailAuthError` (revoked/invalid refresh
token) is terminal — it flips the account to `NEEDS_REAUTH` and notifies the
user instead of retrying, since no amount of retrying fixes a revoked grant.
A single malformed message in `fetch_and_classify_email_task` is logged and
skipped rather than failing the whole poll, matching
`job_ingestion/tasks.py`'s per-posting try/except convention.
"""

from __future__ import annotations

import uuid
from asyncio import run as run_async
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.crypto import decrypt_token, encrypt_token
from app.core.exceptions import ServiceUnavailableError
from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal
from app.integrations.cache import CacheClient
from app.integrations.redis_client import get_redis_client
from app.modules.email_tracking.errors import (
    GmailFailureCategory,
    GmailHistoryExpiredError,
    TransientGmailError,
    classify_gmail_exception,
)
from app.modules.email_tracking.gmail_api_client import GmailAPIClient
from app.modules.email_tracking.gmail_oauth_client import GmailOAuthClient
from app.modules.email_tracking.models import GmailAccount, GmailAccountStatus
from app.modules.email_tracking.repository import GmailAccountRepository, TrackedEmailRepository
from app.modules.email_tracking.service import EmailTrackingService
from app.modules.notification.models import NotificationEventType
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
from app.workers.base_task import BaseTask
from app.workers.celery_app import celery_app

logger = get_logger(__name__)

_settings = get_settings()
_TOKEN_REFRESH_MARGIN_SECONDS = 300


@celery_app.task(base=BaseTask, bind=True, name="email_tracking.sweep_accounts")
def sweep_gmail_accounts_task(self) -> None:  # noqa: ANN001 -- Celery `self`
    run_async(_sweep_accounts())


@celery_app.task(
    base=BaseTask,
    bind=True,
    name="email_tracking.poll_account",
    autoretry_for=(TransientGmailError,),
)
def poll_gmail_account_task(self, account_id: str) -> None:  # noqa: ANN001
    run_async(_poll_account(uuid.UUID(account_id)))


@celery_app.task(
    base=BaseTask,
    bind=True,
    name="email_tracking.fetch_and_classify_email",
    autoretry_for=(TransientGmailError, ServiceUnavailableError),
)
def fetch_and_classify_email_task(self, account_id: str, message_id: str) -> None:  # noqa: ANN001
    run_async(_fetch_and_classify_email(uuid.UUID(account_id), message_id))


def _build_notification_service(session: AsyncSession) -> NotificationService:
    return NotificationService(
        repository=NotificationRepository(session),
        preference_repository=NotificationPreferenceRepository(session),
    )


def _build_email_tracking_service(session: AsyncSession) -> EmailTrackingService:
    tracking_service = TrackingService(
        application_repository=ApplicationRepository(session),
        status_history_repository=ApplicationStatusHistoryRepository(session),
        note_repository=ApplicationNoteRepository(session),
    )
    return EmailTrackingService(
        gmail_account_repository=GmailAccountRepository(session),
        tracked_email_repository=TrackedEmailRepository(session),
        application_repository=ApplicationRepository(session),
        tracking_service=tracking_service,
        notification_service=_build_notification_service(session),
        # Unused in this code path (only OAuth connect/callback touches the
        # cache) — constructed anyway to satisfy the constructor, matching
        # the router's dependency-injection shape.
        cache=CacheClient(get_redis_client()),
    )


async def _sweep_accounts() -> None:
    async with AsyncSessionLocal() as session:
        account_repo = GmailAccountRepository(session)
        accounts = await account_repo.list_connected()
        account_ids = [account.id for account in accounts]

    for account_id in account_ids:
        poll_gmail_account_task.delay(str(account_id))

    logger.info(
        "Gmail sync sweep enqueued", extra={"extra_fields": {"account_count": len(account_ids)}}
    )


async def _ensure_fresh_token(session: AsyncSession, account: GmailAccount) -> str:
    """Returns a valid access token, refreshing it first if it's near expiry."""
    now = datetime.now(UTC)
    if account.token_expires_at - now > timedelta(seconds=_TOKEN_REFRESH_MARGIN_SECONDS):
        return decrypt_token(account.access_token_encrypted)

    oauth_client = GmailOAuthClient()
    tokens = await oauth_client.refresh_access_token(
        refresh_token=decrypt_token(account.refresh_token_encrypted)
    )
    account_repo = GmailAccountRepository(session)
    await account_repo.update(
        account,
        access_token_encrypted=encrypt_token(tokens.access_token),
        token_expires_at=now + timedelta(seconds=tokens.expires_in),
    )
    return tokens.access_token


async def _flag_needs_reauth(
    session: AsyncSession, account: GmailAccount, error: Exception
) -> None:
    account_repo = GmailAccountRepository(session)
    await account_repo.update(
        account, status=GmailAccountStatus.NEEDS_REAUTH, last_sync_error=str(error)[:2000]
    )
    await session.commit()
    notification_service = _build_notification_service(session)
    await notification_service.notify(
        user_id=account.user_id,
        event_type=NotificationEventType.GMAIL_REAUTH_REQUIRED,
        title="Reconnect your Gmail account",
        body=(
            f"We couldn't refresh access to {account.email_address}. Please reconnect it to keep "
            "tracking application emails."
        ),
        data={"gmail_account_id": str(account.id)},
        recipient_email=account.email_address,
    )
    await session.commit()


async def _poll_account(account_id: uuid.UUID) -> None:
    async with AsyncSessionLocal() as session:
        account_repo = GmailAccountRepository(session)
        account = await account_repo.get(account_id)
        if account is None or account.status != GmailAccountStatus.CONNECTED:
            logger.info(
                "Skipping poll for missing/disconnected Gmail account",
                extra={"extra_fields": {"account_id": str(account_id)}},
            )
            return

        try:
            access_token = await _ensure_fresh_token(session, account)
            api_client = GmailAPIClient(access_token=access_token)

            try:
                page = await api_client.list_history(
                    start_history_id=account.history_id or "1",
                    max_results=_settings.GMAIL_MAX_MESSAGES_PER_SYNC,
                )
                message_ids = page.message_ids
                next_history_id = page.next_history_id
            except GmailHistoryExpiredError:
                since = account.last_synced_at or (datetime.now(UTC) - timedelta(days=1))
                message_ids = await api_client.list_message_ids(
                    query=f"after:{int(since.timestamp())} category:primary",
                    max_results=_settings.GMAIL_MAX_MESSAGES_PER_SYNC,
                )
                profile = await api_client.get_profile()
                next_history_id = profile.history_id

            for message_id in message_ids:
                fetch_and_classify_email_task.delay(str(account.id), message_id)

            await account_repo.update(
                account,
                history_id=next_history_id,
                last_synced_at=datetime.now(UTC),
                last_sync_error=None,
            )
            await session.commit()
            logger.info(
                "Gmail account polled",
                extra={
                    "extra_fields": {
                        "account_id": str(account_id),
                        "message_count": len(message_ids),
                    }
                },
            )
        except Exception as exc:
            await session.rollback()
            category = classify_gmail_exception(exc)

            if category == GmailFailureCategory.AUTH:
                await _flag_needs_reauth(session, account, exc)
                logger.warning(
                    "Gmail account needs re-authentication",
                    extra={"extra_fields": {"account_id": str(account_id)}},
                )
                return

            await account_repo.update(account, last_sync_error=str(exc)[:2000])
            await session.commit()
            logger.error(
                "Gmail account poll failed",
                extra={"extra_fields": {"account_id": str(account_id), "error": str(exc)}},
            )
            if category == GmailFailureCategory.TRANSIENT:
                raise TransientGmailError(str(exc)) from exc


async def _fetch_and_classify_email(account_id: uuid.UUID, message_id: str) -> None:
    async with AsyncSessionLocal() as session:
        account_repo = GmailAccountRepository(session)
        account = await account_repo.get(account_id)
        if account is None or account.status != GmailAccountStatus.CONNECTED:
            return

        try:
            access_token = await _ensure_fresh_token(session, account)
            api_client = GmailAPIClient(access_token=access_token)
            message = await api_client.get_message(message_id=message_id)

            service = _build_email_tracking_service(session)
            await service.process_email(account=account, message=message)
            await session.commit()
        except Exception as exc:
            await session.rollback()
            category = classify_gmail_exception(exc)
            logger.error(
                "Failed to fetch/classify a Gmail message — skipping it",
                extra={
                    "extra_fields": {
                        "account_id": str(account_id),
                        "message_id": message_id,
                        "error": str(exc),
                    }
                },
            )
            if category == GmailFailureCategory.AUTH:
                await _flag_needs_reauth(session, account, exc)
                return
            if category == GmailFailureCategory.TRANSIENT:
                raise TransientGmailError(str(exc)) from exc
            # PERMANENT/UNKNOWN: one bad message must not fail the whole poll.


__all__ = ["sweep_gmail_accounts_task", "poll_gmail_account_task", "fetch_and_classify_email_task"]
