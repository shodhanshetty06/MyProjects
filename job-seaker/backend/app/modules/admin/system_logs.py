"""Admin module — "System Logs" cross-module read aggregation.

No log-storage table exists anywhere in the platform (structured logs go to
stdout only, via `app.core.logging`). Per the Phase 12 design decision, this
builds a read-only feed instead of a new write path: it unions the
failure/error signals that already live on disk across modules —
`job_ingestion.IngestionRun`, `auto_apply.AutoApplyAttempt`,
`email_tracking.GmailAccount` sync errors, and `notification.Notification`
send failures — into one timestamp-sorted list. Same "read-only reporting
layer over other modules' tables" convention `analytics` documents itself as
using.

Each source is fetched independently (bounded by `_SOURCE_FETCH_LIMIT`) and
merged/filtered/paginated in Python rather than a single SQL UNION, since the
four source tables have unrelated shapes — acceptable at this data volume and
far simpler than reconciling heterogeneous columns in SQL.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.admin.schemas import SystemLogEntry, SystemLogFilters
from app.modules.auto_apply.models import AutoApplyAttempt, AutoApplyAttemptStatus, AutoApplyTask
from app.modules.email_tracking.models import GmailAccount, GmailAccountStatus
from app.modules.job_ingestion.models import IngestionRun, IngestionRunStatus
from app.modules.notification.models import Notification, NotificationStatus

_SOURCE_FETCH_LIMIT = 500


async def _ingestion_run_entries(session: AsyncSession) -> list[SystemLogEntry]:
    stmt = (
        select(IngestionRun)
        .where(IngestionRun.status.in_([IngestionRunStatus.FAILED, IngestionRunStatus.PARTIAL]))
        .order_by(IngestionRun.created_at.desc())
        .limit(_SOURCE_FETCH_LIMIT)
    )
    runs = (await session.execute(stmt)).scalars().all()
    return [
        SystemLogEntry(
            source="job_ingestion.ingestion_run",
            severity="error" if run.status == IngestionRunStatus.FAILED else "warning",
            message=run.error_message
            or f"Ingestion run for source={run.source} finished as {run.status} "
            f"({run.jobs_failed} jobs failed).",
            target_id=run.company_id,
            occurred_at=run.finished_at or run.created_at,
        )
        for run in runs
    ]


async def _auto_apply_entries(session: AsyncSession) -> list[SystemLogEntry]:
    stmt = (
        select(AutoApplyAttempt, AutoApplyTask.user_id)
        .join(AutoApplyTask, AutoApplyTask.id == AutoApplyAttempt.task_id)
        .where(AutoApplyAttempt.status == AutoApplyAttemptStatus.FAILED)
        .order_by(AutoApplyAttempt.created_at.desc())
        .limit(_SOURCE_FETCH_LIMIT)
    )
    rows = (await session.execute(stmt)).all()
    return [
        SystemLogEntry(
            source="auto_apply.auto_apply_attempt",
            severity="error",
            message=attempt.error_message
            or f"Auto-apply attempt #{attempt.attempt_number} failed"
            f"{f' ({attempt.failure_category})' if attempt.failure_category else ''}.",
            target_id=user_id,
            occurred_at=attempt.finished_at or attempt.created_at,
        )
        for attempt, user_id in rows
    ]


async def _email_tracking_entries(session: AsyncSession) -> list[SystemLogEntry]:
    stmt = (
        select(GmailAccount)
        .where(
            (GmailAccount.status == GmailAccountStatus.NEEDS_REAUTH)
            | (GmailAccount.last_sync_error.is_not(None))
        )
        .order_by(GmailAccount.updated_at.desc())
        .limit(_SOURCE_FETCH_LIMIT)
    )
    accounts = (await session.execute(stmt)).scalars().all()
    return [
        SystemLogEntry(
            source="email_tracking.gmail_account",
            severity="warning" if account.status == GmailAccountStatus.NEEDS_REAUTH else "error",
            message=account.last_sync_error or f"Gmail account needs re-authentication.",
            target_id=account.user_id,
            occurred_at=account.last_synced_at or account.updated_at,
        )
        for account in accounts
    ]


async def _notification_entries(session: AsyncSession) -> list[SystemLogEntry]:
    stmt = (
        select(Notification)
        .where(Notification.status == NotificationStatus.FAILED)
        .order_by(Notification.updated_at.desc())
        .limit(_SOURCE_FETCH_LIMIT)
    )
    notifications = (await session.execute(stmt)).scalars().all()
    return [
        SystemLogEntry(
            source="notification.notification",
            severity="error",
            message=notification.error_message
            or f"Notification delivery failed ({notification.channel}).",
            target_id=notification.user_id,
            occurred_at=notification.updated_at,
        )
        for notification in notifications
    ]


async def collect_system_logs(
    session: AsyncSession, filters: SystemLogFilters
) -> tuple[list[SystemLogEntry], int]:
    entries: list[SystemLogEntry] = []
    for fetch in (
        _ingestion_run_entries,
        _auto_apply_entries,
        _email_tracking_entries,
        _notification_entries,
    ):
        entries.extend(await fetch(session))

    if filters.source:
        entries = [e for e in entries if filters.source in e.source]
    if filters.severity:
        entries = [e for e in entries if e.severity == filters.severity]
    if filters.date_from:
        entries = [e for e in entries if e.occurred_at >= filters.date_from]
    if filters.date_to:
        entries = [e for e in entries if e.occurred_at <= filters.date_to]

    entries.sort(key=lambda e: e.occurred_at, reverse=True)
    total = len(entries)
    page = entries[filters.offset : filters.offset + filters.page_size]
    return page, total


__all__ = ["collect_system_logs"]
