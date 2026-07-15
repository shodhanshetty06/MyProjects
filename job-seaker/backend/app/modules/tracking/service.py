"""Tracking module — application/use-case layer."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from app.core.exceptions import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.modules.tracking.models import (
    Application,
    ApplicationNote,
    ApplicationStatus,
    ApplicationStatusHistory,
)
from app.modules.tracking.repository import (
    ApplicationNoteRepository,
    ApplicationRepository,
    ApplicationStatusHistoryRepository,
)
from app.modules.tracking.schemas import (
    ApplicationCreateRequest,
    ApplicationFilters,
    ApplicationStatistics,
    ApplicationUpdateRequest,
    TimelineEntry,
    TimelineEntryType,
)

logger = get_logger(__name__)

# Applications still `applied` with no activity for this long surface on the
# dashboard as needing follow-up.
_FOLLOW_UP_STALE_DAYS = 7
_RECENT_ACTIVITY_LIMIT = 5
_FOLLOW_UP_LIMIT = 10

# Statuses that count as "progressed past the initial application" for the
# interview-rate statistic.
_INTERVIEW_OR_LATER = (
    ApplicationStatus.INTERVIEW,
    ApplicationStatus.ASSESSMENT,
    ApplicationStatus.OFFER,
    ApplicationStatus.ACCEPTED,
)


class TrackingService:
    """Application CRUD, status transitions, notes, timeline, and statistics."""

    def __init__(
        self,
        application_repository: ApplicationRepository,
        status_history_repository: ApplicationStatusHistoryRepository,
        note_repository: ApplicationNoteRepository,
    ) -> None:
        self.application_repository = application_repository
        self.status_history_repository = status_history_repository
        self.note_repository = note_repository

    # --- CRUD ---------------------------------------------------------------

    async def create_application(
        self, *, user_id: uuid.UUID, payload: ApplicationCreateRequest
    ) -> Application:
        applied_at = payload.applied_at or datetime.now(UTC)
        application = await self.application_repository.create(
            user_id=user_id,
            job_id=payload.job_id,
            company_name=payload.company_name,
            job_title=payload.job_title,
            job_url=payload.job_url,
            location=payload.location,
            source=payload.source,
            status=payload.status,
            applied_at=applied_at,
        )
        await self.status_history_repository.create(
            application_id=application.id,
            from_status=None,
            to_status=payload.status,
            note=None,
        )
        logger.info(
            "Application created",
            extra={
                "extra_fields": {"application_id": str(application.id), "user_id": str(user_id)}
            },
        )
        return await self.get_application(
            user_id=user_id, application_id=application.id, with_details=True
        )

    async def get_application(
        self, *, user_id: uuid.UUID, application_id: uuid.UUID, with_details: bool = False
    ) -> Application:
        application = await self.application_repository.get_owned(
            application_id=application_id, user_id=user_id, with_details=with_details
        )
        if application is None:
            raise NotFoundError("Application not found.")
        return application

    async def list_applications(
        self, *, user_id: uuid.UUID, filters: ApplicationFilters, offset: int, limit: int
    ) -> tuple[Sequence[Application], int]:
        applications = await self.application_repository.list_filtered(
            user_id=user_id, filters=filters, offset=offset, limit=limit
        )
        total = await self.application_repository.count_filtered(user_id=user_id, filters=filters)
        return applications, total

    async def update_application(
        self, *, user_id: uuid.UUID, application_id: uuid.UUID, payload: ApplicationUpdateRequest
    ) -> Application:
        application = await self.get_application(user_id=user_id, application_id=application_id)
        updates = payload.model_dump(exclude_unset=True)
        if updates:
            await self.application_repository.update(application, **updates)
        return await self.get_application(
            user_id=user_id, application_id=application_id, with_details=True
        )

    async def delete_application(self, *, user_id: uuid.UUID, application_id: uuid.UUID) -> None:
        application = await self.get_application(user_id=user_id, application_id=application_id)
        await self.application_repository.update(application, deleted_at=datetime.now(UTC))

    # --- status transitions ---------------------------------------------------

    async def change_status(
        self,
        *,
        user_id: uuid.UUID,
        application_id: uuid.UUID,
        new_status: ApplicationStatus,
        note: str | None,
    ) -> Application:
        application = await self.get_application(user_id=user_id, application_id=application_id)
        previous_status = application.status
        if previous_status == new_status:
            raise ValidationError("Application is already in this status.")
        await self.application_repository.update(application, status=new_status)
        await self.status_history_repository.create(
            application_id=application.id,
            from_status=previous_status,
            to_status=new_status,
            note=note,
        )
        return await self.get_application(
            user_id=user_id, application_id=application_id, with_details=True
        )

    async def get_status_history(
        self, *, user_id: uuid.UUID, application_id: uuid.UUID
    ) -> Sequence[ApplicationStatusHistory]:
        await self.get_application(user_id=user_id, application_id=application_id)
        return await self.status_history_repository.list_by_application(
            application_id=application_id
        )

    # --- notes ------------------------------------------------------------------

    async def add_note(
        self, *, user_id: uuid.UUID, application_id: uuid.UUID, content: str
    ) -> ApplicationNote:
        await self.get_application(user_id=user_id, application_id=application_id)
        return await self.note_repository.create(application_id=application_id, content=content)

    async def list_notes(
        self, *, user_id: uuid.UUID, application_id: uuid.UUID
    ) -> Sequence[ApplicationNote]:
        await self.get_application(user_id=user_id, application_id=application_id)
        return await self.note_repository.list_by_application(application_id=application_id)

    async def update_note(
        self, *, user_id: uuid.UUID, application_id: uuid.UUID, note_id: uuid.UUID, content: str
    ) -> ApplicationNote:
        await self.get_application(user_id=user_id, application_id=application_id)
        note = await self.note_repository.get_for_application(
            note_id=note_id, application_id=application_id
        )
        if note is None:
            raise NotFoundError("Note not found.")
        return await self.note_repository.update(note, content=content)

    async def delete_note(
        self, *, user_id: uuid.UUID, application_id: uuid.UUID, note_id: uuid.UUID
    ) -> None:
        await self.get_application(user_id=user_id, application_id=application_id)
        note = await self.note_repository.get_for_application(
            note_id=note_id, application_id=application_id
        )
        if note is None:
            raise NotFoundError("Note not found.")
        await self.note_repository.delete(note)

    # --- timeline -----------------------------------------------------------------

    async def get_timeline(
        self, *, user_id: uuid.UUID, application_id: uuid.UUID
    ) -> list[TimelineEntry]:
        """Status changes and notes merged into one chronological feed."""
        application = await self.get_application(
            user_id=user_id, application_id=application_id, with_details=True
        )
        entries = [
            TimelineEntry(
                type=TimelineEntryType.STATUS_CHANGE,
                occurred_at=history.created_at,
                data={
                    "id": str(history.id),
                    "from_status": history.from_status,
                    "to_status": history.to_status,
                    "note": history.note,
                },
            )
            for history in application.status_history
        ]
        entries += [
            TimelineEntry(
                type=TimelineEntryType.NOTE,
                occurred_at=note.created_at,
                data={"id": str(note.id), "content": note.content},
            )
            for note in application.notes
        ]
        entries.sort(key=lambda entry: entry.occurred_at)
        return entries

    # --- statistics / dashboard -----------------------------------------------------

    async def get_statistics(self, *, user_id: uuid.UUID) -> ApplicationStatistics:
        status_counts = await self.application_repository.status_counts(user_id=user_id)
        total = sum(status_counts.values())
        now = datetime.now(UTC)
        last_7_days = await self.application_repository.count_since(
            user_id=user_id, since=now - timedelta(days=7)
        )
        last_30_days = await self.application_repository.count_since(
            user_id=user_id, since=now - timedelta(days=30)
        )
        average_days_to_first_response = (
            await self.application_repository.average_days_to_first_response(user_id=user_id)
        )

        interview_plus = sum(status_counts.get(status, 0) for status in _INTERVIEW_OR_LATER)
        offer_plus = status_counts.get(ApplicationStatus.OFFER, 0) + status_counts.get(
            ApplicationStatus.ACCEPTED, 0
        )
        accepted = status_counts.get(ApplicationStatus.ACCEPTED, 0)

        return ApplicationStatistics(
            total_applications=total,
            status_counts=status_counts,
            applications_last_7_days=last_7_days,
            applications_last_30_days=last_30_days,
            interview_rate=round(interview_plus / total, 4) if total else 0.0,
            offer_rate=round(offer_plus / total, 4) if total else 0.0,
            acceptance_rate=round(accepted / offer_plus, 4) if offer_plus else 0.0,
            average_days_to_first_response=average_days_to_first_response,
        )

    async def get_dashboard(
        self, *, user_id: uuid.UUID
    ) -> tuple[ApplicationStatistics, Sequence[Application], Sequence[Application]]:
        statistics = await self.get_statistics(user_id=user_id)
        recent_applications = await self.application_repository.list_recent(
            user_id=user_id, limit=_RECENT_ACTIVITY_LIMIT
        )
        stale_before = datetime.now(UTC) - timedelta(days=_FOLLOW_UP_STALE_DAYS)
        needs_follow_up = await self.application_repository.list_needing_follow_up(
            user_id=user_id, stale_before=stale_before, limit=_FOLLOW_UP_LIMIT
        )
        return statistics, recent_applications, needs_follow_up
