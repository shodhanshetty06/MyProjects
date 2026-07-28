"""Resume Parser module — data access layer."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal, Sequence

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.repositories.base import BaseRepository
from app.modules.resume_parser.models import (
    PipelineRunStatus,
    Resume,
    ResumeParsedData,
    ResumePipelineRun,
    ResumeVersion,
)

# (step name, progress weight) — weights sum to 100. Order matches the
# ResumeUploaded event chain: parse -> embed -> score -> ATS baseline ->
# match -> recommend -> skill gap -> learning recs -> career health ->
# dashboard -> notify. The last three (AI Career Coach insights) require a
# target role: they run once `user_profile.target_roles` is known (usually
# from onboarding completion), see `ai_career_coach.tasks`.
PIPELINE_STEPS: list[tuple[str, int]] = [
    ("parsing", 12),
    ("embedding", 12),
    ("scoring", 8),
    ("ats_baseline", 12),
    ("matching", 18),
    ("recommendations", 8),
    ("skill_gap", 12),
    ("learning_recommendations", 8),
    ("career_health", 8),
    ("dashboard", 1),
    ("notification", 1),
]


class ResumeRepository(BaseRepository[Resume]):
    """Data access for the logical `Resume` entity."""

    model = Resume

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def list_by_user(
        self, *, user_id: uuid.UUID, offset: int = 0, limit: int = 20
    ) -> Sequence[Resume]:
        stmt = (
            select(Resume)
            .where(Resume.user_id == user_id, Resume.deleted_at.is_(None))
            .options(selectinload(Resume.versions))
            .order_by(Resume.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().all()

    async def count_by_user(self, *, user_id: uuid.UUID) -> int:
        stmt = select(func.count()).select_from(Resume).where(
            Resume.user_id == user_id, Resume.deleted_at.is_(None)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def get_owned(self, *, resume_id: uuid.UUID, user_id: uuid.UUID) -> Resume | None:
        stmt = (
            select(Resume)
            .where(Resume.id == resume_id, Resume.user_id == user_id, Resume.deleted_at.is_(None))
            .options(selectinload(Resume.versions))
        )
        result = await self.session.execute(stmt)
        return result.scalars().unique().one_or_none()


class ResumeVersionRepository(BaseRepository[ResumeVersion]):
    """Data access for `ResumeVersion`, including current-version bookkeeping."""

    model = ResumeVersion

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def list_for_resume(self, *, resume_id: uuid.UUID) -> Sequence[ResumeVersion]:
        stmt = (
            select(ResumeVersion)
            .where(ResumeVersion.resume_id == resume_id, ResumeVersion.deleted_at.is_(None))
            .order_by(ResumeVersion.version_number.desc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_current(self, *, resume_id: uuid.UUID) -> ResumeVersion | None:
        stmt = select(ResumeVersion).where(
            ResumeVersion.resume_id == resume_id,
            ResumeVersion.is_current.is_(True),
            ResumeVersion.deleted_at.is_(None),
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def get_for_resume(
        self, *, resume_id: uuid.UUID, version_id: uuid.UUID, with_details: bool = False
    ) -> ResumeVersion | None:
        stmt = select(ResumeVersion).where(
            ResumeVersion.id == version_id,
            ResumeVersion.resume_id == resume_id,
            ResumeVersion.deleted_at.is_(None),
        )
        if with_details:
            stmt = stmt.options(
                selectinload(ResumeVersion.parsed_data), selectinload(ResumeVersion.embedding)
            )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def next_version_number(self, *, resume_id: uuid.UUID) -> int:
        stmt = select(func.max(ResumeVersion.version_number)).where(
            ResumeVersion.resume_id == resume_id
        )
        result = await self.session.execute(stmt)
        current_max = result.scalar_one_or_none()
        return (current_max or 0) + 1

    async def set_current(self, *, resume_id: uuid.UUID, version_id: uuid.UUID) -> None:
        """Atomically make `version_id` the only current version of `resume_id`."""
        await self.session.execute(
            update(ResumeVersion)
            .where(ResumeVersion.resume_id == resume_id, ResumeVersion.id != version_id)
            .values(is_current=False)
        )
        await self.session.execute(
            update(ResumeVersion).where(ResumeVersion.id == version_id).values(is_current=True)
        )
        await self.session.flush()


class ResumeParsedDataRepository(BaseRepository[ResumeParsedData]):
    """Data access for `ResumeParsedData` — the editable AI extraction the
    Profile Review screen lets the user correct before onboarding completes.
    Only the generic `update()` from `BaseRepository` is needed: callers
    reach the row itself through `ResumeVersion.parsed_data` (already
    eager-loaded via `ResumeVersionRepository.get_for_resume(with_details=True)`).
    """

    model = ResumeParsedData

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)


class ResumePipelineRunRepository(BaseRepository[ResumePipelineRun]):
    """Data access for `ResumePipelineRun` — the durable progress tracker
    each task in the auto-orchestrated pipeline updates in place.
    """

    model = ResumePipelineRun

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_by_resume_version(
        self, *, resume_version_id: uuid.UUID
    ) -> ResumePipelineRun | None:
        stmt = select(ResumePipelineRun).where(
            ResumePipelineRun.resume_version_id == resume_version_id
        )
        result = await self.session.execute(stmt)
        return result.scalars().one_or_none()

    async def get_latest_for_user(self, *, user_id: uuid.UUID) -> ResumePipelineRun | None:
        stmt = (
            select(ResumePipelineRun)
            .where(ResumePipelineRun.user_id == user_id)
            .order_by(ResumePipelineRun.created_at.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def start(
        self, *, user_id: uuid.UUID, resume_version_id: uuid.UUID
    ) -> ResumePipelineRun:
        """(Re)start the pipeline for one resume version — a fresh upload of
        a new version gets a fresh run (idempotent: re-uploading twice just
        resets the same row rather than accumulating duplicates).
        """
        steps = [
            {
                "name": name,
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "error": None,
            }
            for name, _ in PIPELINE_STEPS
        ]
        run = await self.get_by_resume_version(resume_version_id=resume_version_id)
        if run is None:
            run = ResumePipelineRun(
                user_id=user_id,
                resume_version_id=resume_version_id,
                status=PipelineRunStatus.PENDING,
                current_step=PIPELINE_STEPS[0][0],
                progress_pct=0,
                steps=steps,
                error_message=None,
            )
            self.session.add(run)
        else:
            run.user_id = user_id
            run.status = PipelineRunStatus.PENDING
            run.current_step = PIPELINE_STEPS[0][0]
            run.progress_pct = 0
            run.steps = steps
            run.error_message = None
        await self.session.flush()
        return run

    async def advance(
        self,
        *,
        resume_version_id: uuid.UUID,
        step: str,
        status: Literal["running", "completed", "failed"],
        error: str | None = None,
    ) -> ResumePipelineRun | None:
        """Update one step's state, recompute `progress_pct` from completed
        step weights, and roll the run-level status up accordingly.
        """
        run = await self.get_by_resume_version(resume_version_id=resume_version_id)
        if run is None:
            return None

        now = datetime.now(timezone.utc).isoformat()
        steps = [dict(s) for s in run.steps]
        for entry in steps:
            if entry["name"] == step:
                entry["status"] = status
                if status == "running":
                    entry["started_at"] = now
                elif status in ("completed", "failed"):
                    entry["completed_at"] = now
                    entry["error"] = error
        run.steps = steps

        weights = dict(PIPELINE_STEPS)
        completed_weight = sum(
            weights.get(s["name"], 0) for s in steps if s["status"] == "completed"
        )
        run.progress_pct = min(100, completed_weight)

        if status == "failed":
            run.status = PipelineRunStatus.FAILED
            run.error_message = error
            run.current_step = step
        elif all(s["status"] == "completed" for s in steps):
            run.status = PipelineRunStatus.COMPLETED
            run.current_step = None
            run.progress_pct = 100
        else:
            run.status = PipelineRunStatus.RUNNING
            run.current_step = step

        await self.session.flush()
        return run
