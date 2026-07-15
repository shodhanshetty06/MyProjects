"""DB-free unit tests for `system_logs.collect_system_logs`'s merge/filter/
paginate logic. The four per-module fetchers are monkeypatched to canned
lists so no Postgres connection is needed — the session argument is never
actually used once they're patched.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.modules.admin import system_logs
from app.modules.admin.schemas import SystemLogEntry, SystemLogFilters

_NOW = datetime(2026, 7, 14, tzinfo=timezone.utc)


def _entry(source: str, severity: str, offset_minutes: int) -> SystemLogEntry:
    return SystemLogEntry(
        source=source,
        severity=severity,
        message=f"{source} event",
        target_id=uuid.uuid4(),
        occurred_at=_NOW - timedelta(minutes=offset_minutes),
    )


@pytest.fixture(autouse=True)
def _patch_sources(monkeypatch: pytest.MonkeyPatch):
    entries = {
        "ingestion": [_entry("job_ingestion.ingestion_run", "error", 10)],
        "auto_apply": [_entry("auto_apply.auto_apply_attempt", "error", 5)],
        "email": [_entry("email_tracking.gmail_account", "warning", 30)],
        "notification": [_entry("notification.notification", "error", 1)],
    }

    async def _ingestion(session):
        return entries["ingestion"]

    async def _auto_apply(session):
        return entries["auto_apply"]

    async def _email(session):
        return entries["email"]

    async def _notification(session):
        return entries["notification"]

    monkeypatch.setattr(system_logs, "_ingestion_run_entries", _ingestion)
    monkeypatch.setattr(system_logs, "_auto_apply_entries", _auto_apply)
    monkeypatch.setattr(system_logs, "_email_tracking_entries", _email)
    monkeypatch.setattr(system_logs, "_notification_entries", _notification)
    yield entries


async def test_collect_merges_all_sources_sorted_by_recency() -> None:
    page, total = await system_logs.collect_system_logs(None, SystemLogFilters())
    assert total == 4
    # Most recent (notification, 1 min ago) first.
    assert page[0].source == "notification.notification"
    assert page[-1].source == "email_tracking.gmail_account"


async def test_collect_filters_by_severity() -> None:
    page, total = await system_logs.collect_system_logs(
        None, SystemLogFilters(severity="warning")
    )
    assert total == 1
    assert page[0].source == "email_tracking.gmail_account"


async def test_collect_filters_by_source_substring() -> None:
    page, total = await system_logs.collect_system_logs(None, SystemLogFilters(source="auto_apply"))
    assert total == 1
    assert page[0].source == "auto_apply.auto_apply_attempt"


async def test_collect_paginates() -> None:
    page, total = await system_logs.collect_system_logs(
        None, SystemLogFilters(page=1, page_size=2)
    )
    assert total == 4
    assert len(page) == 2

    page_2, _ = await system_logs.collect_system_logs(None, SystemLogFilters(page=2, page_size=2))
    assert len(page_2) == 2
    assert {e.source for e in page} != {e.source for e in page_2}
