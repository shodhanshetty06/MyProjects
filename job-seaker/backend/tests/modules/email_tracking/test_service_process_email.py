"""Integration tests for `EmailTrackingService.process_email` — the core
classify -> match -> update-status pipeline, exercised against a real DB
session (same `AsyncSessionLocal` seeding convention used by other modules'
router tests, e.g. `tests/modules/ats_resume_generator/test_router.py`).
Claude classification is monkeypatched (no network/API key required),
matching `tests/modules/resume_parser/test_ai_service.py`'s convention.

Needs a reachable Postgres + Redis with migrations applied:

    docker compose up -d postgres redis
    docker compose exec api alembic upgrade head
    docker compose exec api pytest tests/modules/email_tracking/test_service_process_email.py
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

from app.db.session import AsyncSessionLocal
from app.integrations.cache import CacheClient
from app.integrations.redis_client import get_redis_client
from app.modules.email_tracking.gmail_api_client import GmailMessage
from app.modules.email_tracking.models import EmailMatchStatus, GmailAccount, GmailAccountStatus
from app.modules.email_tracking.repository import GmailAccountRepository, TrackedEmailRepository
from app.modules.email_tracking.service import EmailTrackingService
from app.modules.notification.repository import (
    NotificationPreferenceRepository,
    NotificationRepository,
)
from app.modules.notification.service import NotificationService
from app.modules.tracking.models import Application, ApplicationSource, ApplicationStatus
from app.modules.tracking.repository import (
    ApplicationNoteRepository,
    ApplicationRepository,
    ApplicationStatusHistoryRepository,
)
from app.modules.tracking.service import TrackingService
from app.services.ai.service import AIService


class _FakeAnthropicMessages:
    def __init__(self, response) -> None:
        self._response = response

    async def create(self, **kwargs):
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response) -> None:
        self.messages = _FakeAnthropicMessages(response)


def _classification_response(payload: dict):
    return SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
    )


def _fake_ai_service(payload: dict) -> AIService:
    ai_service = AIService()
    ai_service._anthropic_client = _FakeAnthropicClient(_classification_response(payload))
    return ai_service


async def _seed_gmail_account(*, user_id: uuid.UUID) -> GmailAccount:
    async with AsyncSessionLocal() as session:
        account = GmailAccount(
            user_id=user_id,
            email_address=f"{uuid.uuid4().hex}@example.com",
            status=GmailAccountStatus.CONNECTED,
            access_token_encrypted="placeholder",
            refresh_token_encrypted="placeholder",
            token_expires_at=datetime.now(UTC),
            scopes="https://www.googleapis.com/auth/gmail.readonly",
            history_id="1",
        )
        session.add(account)
        await session.commit()
        await session.refresh(account)
        return account


async def _seed_application(
    *,
    user_id: uuid.UUID,
    company_name: str,
    job_title: str,
    status: ApplicationStatus = ApplicationStatus.APPLIED,
) -> Application:
    async with AsyncSessionLocal() as session:
        application_repo = ApplicationRepository(session)
        history_repo = ApplicationStatusHistoryRepository(session)
        application = await application_repo.create(
            user_id=user_id,
            job_id=None,
            company_name=company_name,
            job_title=job_title,
            job_url=None,
            location=None,
            source=ApplicationSource.MANUAL,
            status=status,
            applied_at=datetime.now(UTC),
        )
        await history_repo.create(
            application_id=application.id, from_status=None, to_status=status, note=None
        )
        await session.commit()
        await session.refresh(application)
        return application


def _message(*, message_id: str, sender: str, subject: str, snippet: str) -> GmailMessage:
    return GmailMessage(
        message_id=message_id,
        thread_id=f"thread-{message_id}",
        sender=sender,
        subject=subject,
        snippet=snippet,
        received_at=datetime.now(UTC),
        label_ids=["INBOX"],
    )


async def _build_service(session, *, ai_service: AIService) -> EmailTrackingService:
    tracking_service = TrackingService(
        application_repository=ApplicationRepository(session),
        status_history_repository=ApplicationStatusHistoryRepository(session),
        note_repository=ApplicationNoteRepository(session),
    )
    notification_service = NotificationService(
        repository=NotificationRepository(session),
        preference_repository=NotificationPreferenceRepository(session),
    )
    return EmailTrackingService(
        gmail_account_repository=GmailAccountRepository(session),
        tracked_email_repository=TrackedEmailRepository(session),
        application_repository=ApplicationRepository(session),
        tracking_service=tracking_service,
        notification_service=notification_service,
        cache=CacheClient(get_redis_client()),
        ai_service=ai_service,
    )


async def test_process_email_updates_application_status_via_tracking_service() -> None:
    user_id = uuid.uuid4()
    account = await _seed_gmail_account(user_id=user_id)
    application = await _seed_application(
        user_id=user_id, company_name="Acme Corp", job_title="Backend Engineer"
    )

    ai_service = _fake_ai_service(
        {
            "is_relevant": True,
            "classification": "interview",
            "confidence": 0.95,
            "extracted_company_name": "Acme Corp",
            "extracted_job_title": "Backend Engineer",
            "reasoning": "Interview scheduling email.",
        }
    )
    message = _message(
        message_id=f"msg-{uuid.uuid4().hex}",
        sender="recruiting@acme-corp.com",
        subject="Interview with Acme Corp",
        snippet="We would like to schedule an interview for the Backend Engineer role.",
    )

    async with AsyncSessionLocal() as session:
        service = await _build_service(session, ai_service=ai_service)
        tracked = await service.process_email(account=account, message=message)
        await session.commit()

    assert tracked is not None
    assert tracked.match_status == EmailMatchStatus.MATCHED
    assert tracked.application_id == application.id
    assert tracked.matched_status_history_id is not None

    async with AsyncSessionLocal() as session:
        application_repo = ApplicationRepository(session)
        refreshed = await application_repo.get_owned(
            application_id=application.id, user_id=user_id, with_details=True
        )
        assert refreshed.status == ApplicationStatus.INTERVIEW
        assert len(refreshed.status_history) == 2  # initial "applied" + this transition


async def test_process_email_swallows_no_op_status_change() -> None:
    user_id = uuid.uuid4()
    account = await _seed_gmail_account(user_id=user_id)
    application = await _seed_application(
        user_id=user_id,
        company_name="Acme Corp",
        job_title="Backend Engineer",
        status=ApplicationStatus.INTERVIEW,
    )

    ai_service = _fake_ai_service(
        {
            "is_relevant": True,
            "classification": "interview",
            "confidence": 0.95,
            "extracted_company_name": "Acme Corp",
            "extracted_job_title": "Backend Engineer",
            "reasoning": "Interview scheduling email.",
        }
    )
    message = _message(
        message_id=f"msg-{uuid.uuid4().hex}",
        sender="recruiting@acme-corp.com",
        subject="Interview with Acme Corp",
        snippet="Following up on your upcoming interview for the Backend Engineer role.",
    )

    async with AsyncSessionLocal() as session:
        service = await _build_service(session, ai_service=ai_service)
        tracked = await service.process_email(account=account, message=message)
        await session.commit()

    assert tracked is not None
    assert tracked.match_status == EmailMatchStatus.MATCHED
    assert tracked.application_id == application.id
    # already-in-this-status is caught and swallowed, not treated as a failure
    assert tracked.matched_status_history_id is None


async def test_process_email_ignores_irrelevant_email() -> None:
    user_id = uuid.uuid4()
    account = await _seed_gmail_account(user_id=user_id)

    ai_service = _fake_ai_service(
        {
            "is_relevant": False,
            "classification": "none",
            "confidence": 0.99,
            "extracted_company_name": None,
            "extracted_job_title": None,
            "reasoning": "Marketing newsletter.",
        }
    )
    message = _message(
        message_id=f"msg-{uuid.uuid4().hex}",
        sender="newsletter@jobsite.com",
        subject="This week's top jobs",
        snippet="Check out these openings.",
    )

    async with AsyncSessionLocal() as session:
        service = await _build_service(session, ai_service=ai_service)
        tracked = await service.process_email(account=account, message=message)
        await session.commit()

    assert tracked is not None
    assert tracked.match_status == EmailMatchStatus.IGNORED
    assert tracked.application_id is None


async def test_process_email_is_idempotent_for_the_same_message() -> None:
    user_id = uuid.uuid4()
    account = await _seed_gmail_account(user_id=user_id)
    await _seed_application(user_id=user_id, company_name="Acme Corp", job_title="Backend Engineer")

    ai_service = _fake_ai_service(
        {
            "is_relevant": True,
            "classification": "rejection",
            "confidence": 0.9,
            "extracted_company_name": "Acme Corp",
            "extracted_job_title": "Backend Engineer",
            "reasoning": "Rejection email.",
        }
    )
    message = _message(
        message_id=f"msg-{uuid.uuid4().hex}",
        sender="recruiting@acme-corp.com",
        subject="Update on your application to Acme Corp",
        snippet="We have decided to move forward with other candidates.",
    )

    async with AsyncSessionLocal() as session:
        service = await _build_service(session, ai_service=ai_service)
        first = await service.process_email(account=account, message=message)
        await session.commit()

    assert first is not None

    async with AsyncSessionLocal() as session:
        service = await _build_service(session, ai_service=ai_service)
        second = await service.process_email(account=account, message=message)
        await session.commit()

    assert second is None
