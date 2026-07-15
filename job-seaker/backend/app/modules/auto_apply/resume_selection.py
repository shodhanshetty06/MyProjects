"""Resume Selection / ATS Resume Selection — picks which resume file to
attach to an application.

Priority, most to least specific:
    1. An explicit override the caller passed in (`ats_resume_id` /
       `resume_version_id` on `AutoApplyTaskCreateRequest`).
    2. A tailored `ATSResume` (`ats_resume_generator`) whose
       `target_job_id` is this exact job.
    3. A tailored `ATSResume` whose `company_name` matches the job's company.
    4. The most recently updated tailored `ATSResume` with a completed,
       downloadable version (generic ATS-optimized resume).
    5. The user's most recently updated plain, parsed `Resume` (not
       ATS-optimized — flagged via `is_ats_optimized=False` so callers can
       nudge the user to generate a tailored one).

Pure given already-loaded ORM collections — no querying here, so it's
cheaply unit-testable with in-memory model instances.
"""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel

from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.modules.ats_resume_generator.models import ATSFileFormat, ATSResume, ATSVersionStatus
from app.modules.job_ingestion.models import Job
from app.modules.resume_parser.models import Resume, ResumeVersionStatus


class ResumeSelection(BaseModel):
    source: str  # "ats_resume" | "resume"
    display_name: str
    is_ats_optimized: bool
    content_type: str
    s3_bucket: str
    s3_key: str
    ats_resume_id: uuid.UUID | None = None
    ats_resume_version_id: uuid.UUID | None = None
    resume_version_id: uuid.UUID | None = None


def _recency_key(item: object) -> datetime:
    return getattr(item, "updated_at", None) or getattr(item, "created_at")


def _sorted_by_recency(items: Sequence[object]) -> list[object]:
    return sorted(items, key=_recency_key, reverse=True)


_CONTENT_TYPES = {
    ATSFileFormat.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ATSFileFormat.PDF: "application/pdf",
}


def _from_ats_resume(
    ats_resume: ATSResume, *, preferred_format: ATSFileFormat
) -> ResumeSelection | None:
    current = next((v for v in ats_resume.versions if v.is_current), None)
    if current is None or current.status != ATSVersionStatus.COMPLETED or not current.files:
        return None

    file_row = next((f for f in current.files if f.format == preferred_format), None) or current.files[0]
    return ResumeSelection(
        source="ats_resume",
        display_name=ats_resume.title,
        is_ats_optimized=True,
        content_type=_CONTENT_TYPES.get(file_row.format, "application/octet-stream"),
        s3_bucket=file_row.s3_bucket,
        s3_key=file_row.s3_key,
        ats_resume_id=ats_resume.id,
        ats_resume_version_id=current.id,
    )


def _from_resume(resume: Resume) -> ResumeSelection | None:
    current = next((v for v in resume.versions if v.is_current), None)
    if current is None or current.status != ResumeVersionStatus.PARSED:
        return None

    content_type = (
        "application/pdf"
        if current.content_type.value == "pdf"
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    return ResumeSelection(
        source="resume",
        display_name=current.original_filename,
        is_ats_optimized=False,
        content_type=content_type,
        s3_bucket=current.s3_bucket,
        s3_key=current.s3_key,
        resume_version_id=current.id,
    )


def select_resume(
    *,
    ats_resumes: Sequence[ATSResume],
    resumes: Sequence[Resume],
    job: Job | None,
    explicit_ats_resume_id: uuid.UUID | None = None,
    explicit_resume_version_id: uuid.UUID | None = None,
    preferred_format: ATSFileFormat = ATSFileFormat.PDF,
) -> ResumeSelection:
    if explicit_ats_resume_id is not None:
        match = next((r for r in ats_resumes if r.id == explicit_ats_resume_id), None)
        if match is None:
            raise NotFoundError("Selected tailored resume not found.")
        selection = _from_ats_resume(match, preferred_format=preferred_format)
        if selection is None:
            raise ConflictError(
                "Selected tailored resume has no completed, downloadable version yet."
            )
        return selection

    if explicit_resume_version_id is not None:
        match = next(
            (
                resume
                for resume in resumes
                if any(v.id == explicit_resume_version_id for v in resume.versions)
            ),
            None,
        )
        if match is None:
            raise NotFoundError("Selected resume version not found.")
        version = next(v for v in match.versions if v.id == explicit_resume_version_id)
        if version.status != ResumeVersionStatus.PARSED:
            raise ConflictError("Selected resume version hasn't finished parsing yet.")
        content_type = (
            "application/pdf"
            if version.content_type.value == "pdf"
            else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        return ResumeSelection(
            source="resume",
            display_name=version.original_filename,
            is_ats_optimized=False,
            content_type=content_type,
            s3_bucket=version.s3_bucket,
            s3_key=version.s3_key,
            resume_version_id=version.id,
        )

    if job is not None:
        exact_matches = [r for r in ats_resumes if r.target_job_id == job.id]
        for candidate in _sorted_by_recency(exact_matches):
            selection = _from_ats_resume(candidate, preferred_format=preferred_format)  # type: ignore[arg-type]
            if selection is not None:
                return selection

        company = getattr(job, "company", None)
        if company is not None and company.name:
            company_matches = [
                r
                for r in ats_resumes
                if r.company_name and r.company_name.strip().lower() == company.name.strip().lower()
            ]
            for candidate in _sorted_by_recency(company_matches):
                selection = _from_ats_resume(candidate, preferred_format=preferred_format)  # type: ignore[arg-type]
                if selection is not None:
                    return selection

    for candidate in _sorted_by_recency(list(ats_resumes)):
        selection = _from_ats_resume(candidate, preferred_format=preferred_format)  # type: ignore[arg-type]
        if selection is not None:
            return selection

    for candidate in _sorted_by_recency(list(resumes)):
        selection = _from_resume(candidate)  # type: ignore[arg-type]
        if selection is not None:
            return selection

    raise ValidationError(
        "No usable resume is available — upload and parse a resume (or generate a tailored "
        "ATS resume) before auto-applying.",
        error_code="no_resume_available",
    )


__all__ = ["ResumeSelection", "select_resume"]
