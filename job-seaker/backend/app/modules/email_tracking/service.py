"""Email Tracking module — application/use-case layer.

Owns the Gmail OAuth connect/disconnect flow, manual review resolution, and
the classify -> match -> update-status pipeline (`process_email`, called
from `tasks.py`). Kept independent of Celery so `process_email` is directly
unit-testable without any task-queue machinery.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from app.core.config import get_settings
from app.core.crypto import decrypt_token, encrypt_token
from app.core.exceptions import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.dependency_injection.providers import build_ai_service
from app.integrations.cache import CacheClient
from app.modules.email_tracking.classifier import classify_email
from app.modules.email_tracking.gmail_api_client import GmailAPIClient, GmailMessage
from app.modules.email_tracking.gmail_oauth_client import GmailOAuthClient
from app.modules.email_tracking.matching import match_application
from app.modules.email_tracking.models import (
    EmailClassification,
    EmailMatchStatus,
    GmailAccount,
    GmailAccountStatus,
    TrackedEmail,
)
from app.modules.email_tracking.repository import GmailAccountRepository, TrackedEmailRepository
from app.modules.email_tracking.schemas import TrackedEmailFilters, TrackedEmailResolveRequest
from app.modules.notification.models import NotificationEventType
from app.modules.notification.service import NotificationService
from app.modules.tracking.models import ApplicationStatus
from app.modules.tracking.repository import ApplicationRepository
from app.modules.tracking.schemas import ApplicationFilters
from app.modules.tracking.service import TrackingService
from app.services.ai.service import AIService

logger = get_logger(__name__)

_STATE_CACHE_NAMESPACE = "gmail_oauth_state"
_MATCH_CANDIDATE_LIMIT = 500

_CLASSIFICATION_TO_STATUS = {
    EmailClassification.INTERVIEW: ApplicationStatus.INTERVIEW,
    EmailClassification.ASSESSMENT: ApplicationStatus.ASSESSMENT,
    EmailClassification.REJECTION: ApplicationStatus.REJECTED,
    EmailClassification.OFFER: ApplicationStatus.OFFER,
}


def _state_key(state: str) -> str:
    return f"{_STATE_CACHE_NAMESPACE}:{state}"


class EmailTrackingService:
    def __init__(
        self,
        *,
        gmail_account_repository: GmailAccountRepository,
        tracked_email_repository: TrackedEmailRepository,
        application_repository: ApplicationRepository,
        tracking_service: TrackingService,
        notification_service: NotificationService,
        cache: CacheClient,
        ai_service: AIService | None = None,
        oauth_client: GmailOAuthClient | None = None,
    ) -> None:
        self.gmail_account_repository = gmail_account_repository
        self.tracked_email_repository = tracked_email_repository
        self.application_repository = application_repository
        self.tracking_service = tracking_service
        self.notification_service = notification_service
        self.cache = cache
        self.settings = get_settings()
        self.ai_service = ai_service or build_ai_service(self.settings)
        self.oauth_client = oauth_client or GmailOAuthClient()

    # --- OAuth connect/disconnect -------------------------------------------

    async def start_connect(self, *, user_id: uuid.UUID) -> str:
        state = uuid.uuid4().hex
        await self.cache.set_json(
            _state_key(state),
            {"user_id": str(user_id)},
            ttl_seconds=self.settings.GMAIL_OAUTH_STATE_TTL_SECONDS,
        )
        return self.oauth_client.build_authorization_url(state=state)

    async def complete_connect(self, *, code: str, state: str) -> GmailAccount:
        payload = await self.cache.get_json(_state_key(state))
        if payload is None:
            raise ValidationError("OAuth state is invalid or has expired.")
        user_id = uuid.UUID(payload["user_id"])
        await self.cache.delete(_state_key(state))

        tokens = await self.oauth_client.exchange_code_for_tokens(code=code)
        api_client = GmailAPIClient(access_token=tokens.access_token)
        profile = await api_client.get_profile()
        expires_at = datetime.now(UTC) + timedelta(seconds=tokens.expires_in)

        existing = await self.gmail_account_repository.get_by_email(
            email_address=profile.email_address
        )
        if existing is not None:
            refresh_token = tokens.refresh_token or decrypt_token(existing.refresh_token_encrypted)
            account = await self.gmail_account_repository.update(
                existing,
                user_id=user_id,
                status=GmailAccountStatus.CONNECTED,
                access_token_encrypted=encrypt_token(tokens.access_token),
                refresh_token_encrypted=encrypt_token(refresh_token),
                token_expires_at=expires_at,
                scopes=tokens.scope,
                history_id=profile.history_id,
                last_sync_error=None,
            )
        else:
            if not tokens.refresh_token:
                raise ValidationError(
                    "Google did not return a refresh token. Revoke this app's prior access at "
                    "https://myaccount.google.com/permissions and reconnect."
                )
            account = await self.gmail_account_repository.create(
                user_id=user_id,
                email_address=profile.email_address,
                status=GmailAccountStatus.CONNECTED,
                access_token_encrypted=encrypt_token(tokens.access_token),
                refresh_token_encrypted=encrypt_token(tokens.refresh_token),
                token_expires_at=expires_at,
                scopes=tokens.scope,
                history_id=profile.history_id,
            )

        logger.info(
            "Gmail account connected", extra={"extra_fields": {"account_id": str(account.id)}}
        )
        return account

    async def disconnect(self, *, user_id: uuid.UUID, account_id: uuid.UUID) -> None:
        account = await self.get_owned_account(user_id=user_id, account_id=account_id)
        await self.oauth_client.revoke_token(token=decrypt_token(account.refresh_token_encrypted))
        await self.gmail_account_repository.update(
            account,
            status=GmailAccountStatus.DISCONNECTED,
            access_token_encrypted="",
            refresh_token_encrypted="",
        )

    async def get_owned_account(self, *, user_id: uuid.UUID, account_id: uuid.UUID) -> GmailAccount:
        account = await self.gmail_account_repository.get_owned(
            account_id=account_id, user_id=user_id
        )
        if account is None:
            raise NotFoundError("Gmail account not found.")
        return account

    async def list_accounts(self, *, user_id: uuid.UUID) -> Sequence[GmailAccount]:
        return await self.gmail_account_repository.list_by_user(user_id=user_id)

    # --- tracked emails: listing + manual review -----------------------------

    async def list_tracked_emails(
        self, *, user_id: uuid.UUID, filters: TrackedEmailFilters, offset: int, limit: int
    ) -> tuple[Sequence[TrackedEmail], int]:
        emails = await self.tracked_email_repository.list_filtered(
            user_id=user_id, filters=filters, offset=offset, limit=limit
        )
        total = await self.tracked_email_repository.count_filtered(user_id=user_id, filters=filters)
        return emails, total

    async def get_tracked_email(self, *, user_id: uuid.UUID, email_id: uuid.UUID) -> TrackedEmail:
        email = await self.tracked_email_repository.get_owned(email_id=email_id, user_id=user_id)
        if email is None:
            raise NotFoundError("Tracked email not found.")
        return email

    async def resolve_tracked_email(
        self, *, user_id: uuid.UUID, email_id: uuid.UUID, payload: TrackedEmailResolveRequest
    ) -> TrackedEmail:
        email = await self.get_tracked_email(user_id=user_id, email_id=email_id)
        if payload.dismiss:
            return await self.tracked_email_repository.update(
                email, match_status=EmailMatchStatus.IGNORED
            )

        assert payload.application_id is not None and payload.status is not None
        matched_status_history_id = None
        try:
            application = await self.tracking_service.change_status(
                user_id=user_id,
                application_id=payload.application_id,
                new_status=payload.status,
                note=f'Confirmed from email "{email.subject}" during manual review.',
            )
            matched_status_history_id = application.status_history[-1].id
        except ValidationError:
            pass  # already in this status — nothing further to record

        return await self.tracked_email_repository.update(
            email,
            match_status=EmailMatchStatus.MATCHED,
            application_id=payload.application_id,
            matched_status_history_id=matched_status_history_id,
        )

    # --- classify -> match -> update-status pipeline ---------------------------

    async def process_email(
        self, *, account: GmailAccount, message: GmailMessage
    ) -> TrackedEmail | None:
        """Called per-message from `tasks.py::fetch_and_classify_email_task`.

        Returns `None` if the message was already processed (idempotency
        guard against re-listing the same range after a crash mid-sweep).
        """
        existing = await self.tracked_email_repository.get_by_message_id(
            gmail_account_id=account.id, gmail_message_id=message.message_id
        )
        if existing is not None:
            return None

        result = await classify_email(
            self.ai_service,
            sender=message.sender,
            subject=message.subject,
            snippet=message.snippet,
        )

        if not result.is_relevant or result.classification == "none":
            return await self._persist(
                account=account,
                message=message,
                classification=EmailClassification.NOT_RELEVANT,
                confidence=result.confidence,
                match_status=EmailMatchStatus.IGNORED,
                application_id=None,
                matched_status_history_id=None,
                raw_metadata=result.model_dump(),
            )

        classification = EmailClassification(result.classification)

        if result.confidence < self.settings.EMAIL_TRACKING_CLASSIFICATION_CONFIDENCE_THRESHOLD:
            tracked = await self._persist(
                account=account,
                message=message,
                classification=classification,
                confidence=result.confidence,
                match_status=EmailMatchStatus.NEEDS_REVIEW,
                application_id=None,
                matched_status_history_id=None,
                raw_metadata=result.model_dump(),
            )
            await self._notify_review_needed(account=account, email=tracked)
            return tracked

        lookback_since = datetime.now(UTC) - timedelta(
            days=self.settings.EMAIL_TRACKING_MATCH_LOOKBACK_DAYS
        )
        candidates = await self.application_repository.list_filtered(
            user_id=account.user_id,
            filters=ApplicationFilters(applied_after=lookback_since),
            offset=0,
            limit=_MATCH_CANDIDATE_LIMIT,
        )
        match = match_application(
            candidates=candidates,
            sender=message.sender,
            subject=message.subject,
            snippet=message.snippet,
            extracted_company_name=result.extracted_company_name,
            extracted_job_title=result.extracted_job_title,
            company_weight=self.settings.EMAIL_TRACKING_COMPANY_MATCH_WEIGHT,
            title_weight=self.settings.EMAIL_TRACKING_TITLE_MATCH_WEIGHT,
            confidence_threshold=self.settings.EMAIL_TRACKING_MATCH_CONFIDENCE_THRESHOLD,
            margin=self.settings.EMAIL_TRACKING_MATCH_MARGIN,
        )

        if not match.is_confident:
            has_viable_guess = (
                match.application is not None
                and match.score >= self.settings.EMAIL_TRACKING_MATCH_LOW_SCORE_FLOOR
            )
            tracked = await self._persist(
                account=account,
                message=message,
                classification=classification,
                confidence=result.confidence,
                match_status=(
                    EmailMatchStatus.NEEDS_REVIEW
                    if has_viable_guess
                    else EmailMatchStatus.UNMATCHED
                ),
                application_id=(
                    match.application.id
                    if has_viable_guess and match.application is not None
                    else None
                ),
                matched_status_history_id=None,
                raw_metadata=result.model_dump(),
            )
            await self._notify_review_needed(account=account, email=tracked)
            return tracked

        assert match.application is not None
        new_status = _CLASSIFICATION_TO_STATUS[classification]
        matched_status_history_id = None
        try:
            application = await self.tracking_service.change_status(
                user_id=account.user_id,
                application_id=match.application.id,
                new_status=new_status,
                note=(
                    f'Detected from email "{message.subject}" '
                    f"(received {message.received_at.isoformat()})."
                ),
            )
            matched_status_history_id = application.status_history[-1].id
            # `TrackingService.change_status` now sends the
            # APPLICATION_STATUS_CHANGED notification itself for every
            # status change app-wide (see its Smart Trigger hook) — no
            # separate notify call needed here anymore.
        except ValidationError:
            pass  # application already in this status — not an error, nothing to notify

        return await self._persist(
            account=account,
            message=message,
            classification=classification,
            confidence=result.confidence,
            match_status=EmailMatchStatus.MATCHED,
            application_id=match.application.id,
            matched_status_history_id=matched_status_history_id,
            raw_metadata=result.model_dump(),
        )

    async def _persist(
        self,
        *,
        account: GmailAccount,
        message: GmailMessage,
        classification: EmailClassification,
        confidence: float,
        match_status: EmailMatchStatus,
        application_id: uuid.UUID | None,
        matched_status_history_id: uuid.UUID | None,
        raw_metadata: dict,
    ) -> TrackedEmail:
        return await self.tracked_email_repository.create(
            gmail_account_id=account.id,
            application_id=application_id,
            gmail_message_id=message.message_id,
            thread_id=message.thread_id,
            sender=message.sender,
            subject=message.subject,
            snippet=message.snippet,
            received_at=message.received_at,
            classification=classification,
            confidence=confidence,
            match_status=match_status,
            matched_status_history_id=matched_status_history_id,
            raw_metadata=raw_metadata,
        )

    async def _notify_review_needed(self, *, account: GmailAccount, email: TrackedEmail) -> None:
        await self.notification_service.notify(
            user_id=account.user_id,
            event_type=NotificationEventType.EMAIL_REVIEW_NEEDED,
            title="An email needs your review",
            body=(
                f'We found an email ("{email.subject}") that may relate to a job application '
                "but need your confirmation."
            ),
            data={"tracked_email_id": str(email.id)},
            recipient_email=account.email_address,
        )


__all__ = ["EmailTrackingService"]
