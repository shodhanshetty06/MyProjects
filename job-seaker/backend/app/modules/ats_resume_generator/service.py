"""ATS Resume Generator module — application/use-case layer.

Orchestrates resume selection, version/file bookkeeping, and diff/download
lookups. The actual Claude optimize pass and DOCX/PDF generation happen on
Celery workers (see `tasks.py`) — this layer only validates + enqueues them
and never calls the AI service directly, mirroring
`resume_parser.ResumeParserService`.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Sequence

from app.core.config import get_settings
from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.core.logging import get_logger
from app.integrations.s3_client import S3Storage
from app.modules.ats_resume_generator import diffing
from app.modules.ats_resume_generator.models import (
    ATSFileFormat,
    ATSResume,
    ATSResumeVersion,
    ATSVersionStatus,
)
from app.modules.ats_resume_generator.repository import (
    ATSResumeRepository,
    ATSResumeVersionRepository,
)
from app.modules.ats_resume_generator.schemas import (
    ATSResumeCreateRequest,
    ATSResumeVersionCreateRequest,
    ATSResumeVersionDiff,
    ListDiffSchema,
    ProjectDiffSchema,
    TextDiffSegmentSchema,
)
from app.modules.resume_parser.models import ResumeVersion, ResumeVersionStatus

logger = get_logger(__name__)


class ATSResumeGeneratorService:
    """Business logic orchestration for the ATS Resume Generator module."""

    def __init__(
        self,
        ats_resume_repository: ATSResumeRepository,
        version_repository: ATSResumeVersionRepository,
        s3_storage: S3Storage,
    ) -> None:
        self.ats_resume_repository = ats_resume_repository
        self.version_repository = version_repository
        self.s3_storage = s3_storage
        self.settings = get_settings()

    async def _resolve_source_version(
        self, *, user_id: uuid.UUID, resume_version_id: uuid.UUID
    ) -> ResumeVersion:
        source = await self.version_repository.get_source_resume_version_owned(
            resume_version_id=resume_version_id, user_id=user_id
        )
        if source is None:
            raise NotFoundError("Source resume version not found.")
        if source.status != ResumeVersionStatus.PARSED or source.parsed_data is None:
            raise ConflictError(
                "This resume version hasn't finished parsing yet — optimization needs its "
                "extracted summary/skills/experience/projects first."
            )
        return source

    # --- create ----------------------------------------------------------------------

    async def create_ats_resume(
        self, *, user_id: uuid.UUID, payload: ATSResumeCreateRequest
    ) -> tuple[ATSResume, ATSResumeVersion]:
        source = await self._resolve_source_version(
            user_id=user_id, resume_version_id=payload.resume_version_id
        )
        if payload.target_job_id is not None and not await self.version_repository.job_exists(
            job_id=payload.target_job_id
        ):
            raise NotFoundError("Target job not found.")

        ats_resume = await self.ats_resume_repository.create(
            user_id=user_id,
            source_resume_id=source.resume_id,
            target_job_id=payload.target_job_id,
            title=payload.title or f"{source.original_filename} (ATS)",
            job_title=payload.job_title,
            company_name=payload.company_name,
        )
        version = await self._create_version(
            ats_resume=ats_resume,
            source=source,
            job_description=payload.job_description,
            version_number=1,
        )
        await self.version_repository.set_current(ats_resume_id=ats_resume.id, version_id=version.id)

        logger.info(
            "ATS resume created",
            extra={"extra_fields": {"ats_resume_id": str(ats_resume.id), "user_id": str(user_id)}},
        )
        self._enqueue_optimize(version.id)
        return ats_resume, version

    async def create_new_version(
        self, *, user_id: uuid.UUID, ats_resume_id: uuid.UUID, payload: ATSResumeVersionCreateRequest
    ) -> ATSResumeVersion:
        ats_resume = await self.get_ats_resume(user_id=user_id, ats_resume_id=ats_resume_id)
        current = next((v for v in ats_resume.versions if v.is_current), None)
        if current is None:
            raise ConflictError("This ATS resume has no current version to base a new one on.")

        resume_version_id = payload.resume_version_id or current.source_resume_version_id
        job_description = payload.job_description or current.job_description
        source = await self._resolve_source_version(
            user_id=user_id, resume_version_id=resume_version_id
        )

        version_number = await self.version_repository.next_version_number(
            ats_resume_id=ats_resume.id
        )
        version = await self._create_version(
            ats_resume=ats_resume,
            source=source,
            job_description=job_description,
            version_number=version_number,
        )
        await self.version_repository.set_current(ats_resume_id=ats_resume.id, version_id=version.id)

        logger.info(
            "ATS resume version created",
            extra={
                "extra_fields": {
                    "ats_resume_id": str(ats_resume.id),
                    "version_id": str(version.id),
                    "version_number": version_number,
                }
            },
        )
        self._enqueue_optimize(version.id)
        return version

    async def _create_version(
        self,
        *,
        ats_resume: ATSResume,
        source: ResumeVersion,
        job_description: str,
        version_number: int,
    ) -> ATSResumeVersion:
        parsed = source.parsed_data
        assert parsed is not None  # guaranteed by `_resolve_source_version`
        return await self.version_repository.create(
            ats_resume_id=ats_resume.id,
            version_number=version_number,
            is_current=False,
            status=ATSVersionStatus.PENDING,
            source_resume_version_id=source.id,
            job_description=job_description,
            experience=parsed.experience,
            education=parsed.education,
            certifications=parsed.certifications,
            languages=parsed.languages,
            achievements=parsed.achievements,
            original_summary=parsed.summary,
            original_skills=parsed.skills,
            original_projects=parsed.projects,
        )

    def _enqueue_optimize(self, version_id: uuid.UUID) -> None:
        from app.modules.ats_resume_generator.tasks import optimize_ats_resume_task

        optimize_ats_resume_task.delay(str(version_id))

    # --- reads -----------------------------------------------------------------------

    async def list_ats_resumes(
        self, *, user_id: uuid.UUID, offset: int, limit: int
    ) -> tuple[Sequence[ATSResume], int]:
        resumes = await self.ats_resume_repository.list_by_user(
            user_id=user_id, offset=offset, limit=limit
        )
        total = await self.ats_resume_repository.count_by_user(user_id=user_id)
        return resumes, total

    async def get_ats_resume(self, *, user_id: uuid.UUID, ats_resume_id: uuid.UUID) -> ATSResume:
        ats_resume = await self.ats_resume_repository.get_owned(
            ats_resume_id=ats_resume_id, user_id=user_id
        )
        if ats_resume is None:
            raise NotFoundError("ATS resume not found.")
        return ats_resume

    async def get_version(
        self, *, user_id: uuid.UUID, ats_resume_id: uuid.UUID, version_id: uuid.UUID
    ) -> ATSResumeVersion:
        await self.get_ats_resume(user_id=user_id, ats_resume_id=ats_resume_id)
        version = await self.version_repository.get_for_resume(
            ats_resume_id=ats_resume_id, version_id=version_id
        )
        if version is None:
            raise NotFoundError("ATS resume version not found.")
        return version

    async def list_versions(
        self, *, user_id: uuid.UUID, ats_resume_id: uuid.UUID
    ) -> Sequence[ATSResumeVersion]:
        await self.get_ats_resume(user_id=user_id, ats_resume_id=ats_resume_id)
        return await self.version_repository.list_for_resume(ats_resume_id=ats_resume_id)

    # --- update / delete ---------------------------------------------------------------

    async def update_ats_resume(
        self, *, user_id: uuid.UUID, ats_resume_id: uuid.UUID, title: str
    ) -> ATSResume:
        ats_resume = await self.get_ats_resume(user_id=user_id, ats_resume_id=ats_resume_id)
        return await self.ats_resume_repository.update(ats_resume, title=title)

    async def delete_ats_resume(self, *, user_id: uuid.UUID, ats_resume_id: uuid.UUID) -> None:
        ats_resume = await self.get_ats_resume(user_id=user_id, ats_resume_id=ats_resume_id)
        await self.ats_resume_repository.update(ats_resume, deleted_at=datetime.now(timezone.utc))

    async def delete_version(
        self, *, user_id: uuid.UUID, ats_resume_id: uuid.UUID, version_id: uuid.UUID
    ) -> None:
        version = await self.get_version(
            user_id=user_id, ats_resume_id=ats_resume_id, version_id=version_id
        )
        remaining = await self.version_repository.list_for_resume(ats_resume_id=ats_resume_id)
        if len(remaining) <= 1:
            raise ConflictError("Cannot delete the only version of an ATS resume.")

        was_current = version.is_current
        await self.version_repository.update(version, deleted_at=datetime.now(timezone.utc))

        if was_current:
            candidates = [v for v in remaining if v.id != version.id]
            newest = max(candidates, key=lambda v: v.version_number)
            await self.version_repository.set_current(
                ats_resume_id=ats_resume_id, version_id=newest.id
            )

    async def activate_version(
        self, *, user_id: uuid.UUID, ats_resume_id: uuid.UUID, version_id: uuid.UUID
    ) -> ATSResumeVersion:
        version = await self.get_version(
            user_id=user_id, ats_resume_id=ats_resume_id, version_id=version_id
        )
        await self.version_repository.set_current(ats_resume_id=ats_resume_id, version_id=version_id)
        version.is_current = True
        return version

    # --- downloads ---------------------------------------------------------------------

    async def get_download_url(
        self,
        *,
        user_id: uuid.UUID,
        ats_resume_id: uuid.UUID,
        version_id: uuid.UUID,
        format: ATSFileFormat,
    ) -> str:
        version = await self.get_version(
            user_id=user_id, ats_resume_id=ats_resume_id, version_id=version_id
        )
        file_row = next((f for f in version.files if f.format == format), None)
        if file_row is None:
            raise NotFoundError(
                f"No {format.value.upper()} file has been generated for this version yet."
            )
        return self.s3_storage.generate_presigned_download_url(file_row.s3_key)

    # --- diff view ---------------------------------------------------------------------

    async def get_diff(
        self,
        *,
        user_id: uuid.UUID,
        ats_resume_id: uuid.UUID,
        version_id: uuid.UUID,
        compare_to: str,
    ) -> ATSResumeVersionDiff:
        version = await self.get_version(
            user_id=user_id, ats_resume_id=ats_resume_id, version_id=version_id
        )

        if compare_to == "source":
            base_summary = version.original_summary
            base_skills = version.original_skills
            base_projects = version.original_projects
        else:
            try:
                base_version_id = uuid.UUID(compare_to)
            except ValueError as exc:
                raise ValidationError("compare_to must be 'source' or a version id.") from exc
            base_version = await self.get_version(
                user_id=user_id, ats_resume_id=ats_resume_id, version_id=base_version_id
            )
            base_summary = base_version.optimized_summary
            base_skills = base_version.optimized_skills
            base_projects = base_version.optimized_projects

        summary_diff = diffing.diff_text(base_summary, version.optimized_summary)
        skills_diff = diffing.diff_string_list(base_skills, version.optimized_skills)
        projects_diff = diffing.diff_projects(base_projects, version.optimized_projects)

        return ATSResumeVersionDiff(
            compare_to=compare_to,
            summary_diff=[TextDiffSegmentSchema(op=s.op, text=s.text) for s in summary_diff],
            skills_diff=ListDiffSchema(
                added=skills_diff.added,
                removed=skills_diff.removed,
                unchanged=skills_diff.unchanged,
            ),
            projects_diff=[
                ProjectDiffSchema(
                    name=p.name,
                    status=p.status,
                    description_diff=[
                        TextDiffSegmentSchema(op=s.op, text=s.text) for s in p.description_diff
                    ],
                    highlights_diff=ListDiffSchema(
                        added=p.highlights_diff.added,
                        removed=p.highlights_diff.removed,
                        unchanged=p.highlights_diff.unchanged,
                    ),
                )
                for p in projects_diff
            ],
        )


__all__ = ["ATSResumeGeneratorService"]
