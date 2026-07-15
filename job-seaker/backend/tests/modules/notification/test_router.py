"""Integration tests for the `/notifications` API — list/read/preferences.

Needs a reachable Postgres with the notification tables migrated:

    docker compose up -d postgres redis
    docker compose exec api alembic upgrade head
    docker compose exec api pytest tests/modules/notification/test_router.py

Notifications aren't created via a public endpoint (only other modules call
`NotificationService.notify` internally), so tests seed rows directly via
`AsyncSessionLocal`, following the DB-seeding convention used by
`tests/modules/ats_resume_generator/test_router.py`.
"""

from __future__ import annotations

import uuid

from httpx import AsyncClient

from app.core.security import create_access_token
from app.db.session import AsyncSessionLocal
from app.modules.notification.models import NotificationEventType
from app.modules.notification.repository import (
    NotificationPreferenceRepository,
    NotificationRepository,
)
from app.modules.notification.service import NotificationService


def _user_header(user_id: uuid.UUID | None = None) -> dict[str, str]:
    token, _ = create_access_token(user_id or uuid.uuid4())
    return {"Authorization": f"Bearer {token}"}


async def _seed_notifications(*, user_id: uuid.UUID, count: int = 2) -> None:
    async with AsyncSessionLocal() as session:
        service = NotificationService(
            repository=NotificationRepository(session),
            preference_repository=NotificationPreferenceRepository(session),
        )
        for i in range(count):
            await service.notify(
                user_id=user_id,
                event_type=NotificationEventType.EMAIL_REVIEW_NEEDED,
                title=f"Review needed #{i}",
                body="An email needs your review.",
                recipient_email=None,
            )
        await session.commit()


async def test_list_notifications_returns_only_the_caller_s_notifications(
    client: AsyncClient,
) -> None:
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    await _seed_notifications(user_id=user_a, count=2)
    await _seed_notifications(user_id=user_b, count=1)

    response = await client.get("/api/v1/notifications", headers=_user_header(user_a))

    assert response.status_code == 200
    body = response.json()
    assert body["pagination"]["total_items"] == 2


async def test_endpoints_require_user_id_header(client: AsyncClient) -> None:
    response = await client.get("/api/v1/notifications")
    assert response.status_code == 401


async def test_mark_read_then_unread_only_filter_excludes_it(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    await _seed_notifications(user_id=user_id, count=1)
    headers = _user_header(user_id)

    listed = await client.get("/api/v1/notifications", headers=headers)
    notification_id = listed.json()["data"][0]["id"]

    read_response = await client.post(
        f"/api/v1/notifications/{notification_id}/read", headers=headers
    )
    assert read_response.status_code == 200
    assert read_response.json()["data"]["read_at"] is not None

    unread = await client.get(
        "/api/v1/notifications", params={"unread_only": True}, headers=headers
    )
    assert unread.json()["pagination"]["total_items"] == 0


async def test_mark_all_read(client: AsyncClient) -> None:
    user_id = uuid.uuid4()
    await _seed_notifications(user_id=user_id, count=3)
    headers = _user_header(user_id)

    response = await client.post("/api/v1/notifications/read-all", headers=headers)

    assert response.status_code == 200
    assert response.json()["data"]["marked_read"] == 3


async def test_get_preferences_returns_full_matrix(client: AsyncClient) -> None:
    headers = _user_header()

    response = await client.get("/api/v1/notifications/preferences", headers=headers)

    assert response.status_code == 200
    preferences = response.json()["data"]
    assert len(preferences) > 0
    assert {"event_type", "channel", "enabled"} <= set(preferences[0].keys())


async def test_update_preferences_persists(client: AsyncClient) -> None:
    headers = _user_header()
    payload = {
        "preferences": [
            {"event_type": "application_status_changed", "channel": "in_app", "enabled": False}
        ]
    }

    update_response = await client.put(
        "/api/v1/notifications/preferences", headers=headers, json=payload
    )
    assert update_response.status_code == 200

    get_response = await client.get("/api/v1/notifications/preferences", headers=headers)
    updated = next(
        p
        for p in get_response.json()["data"]
        if p["event_type"] == "application_status_changed" and p["channel"] == "in_app"
    )
    assert updated["enabled"] is False
