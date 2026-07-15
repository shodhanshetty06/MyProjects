"""Unit tests for Resume Selection / ATS Resume Selection."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from app.core.exceptions import ValidationError
from app.modules.ats_resume_generator.models import (
    ATSFileFormat,
    ATSResume,
    ATSResumeFile,
    ATSResumeVersion,
    ATSVersionStatus,
)
from app.modules.auto_apply.resume_selection import select_resume
from app.modules.job_ingestion.models import Company, Job
from app.modules.resume_parser.models import Resume, ResumeContentType, ResumeVersion, ResumeVersionStatus

_NOW = datetime.now(UTC)


def _ts(minutes_ago: int) -> datetime:
    return _NOW - timedelta(minutes=minutes_ago)


def _ats_resume(
    *, target_job_id=None, company_name=None, minutes_ago: int = 0, completed: bool = True, title="Tailored"
) -> ATSResume:
    version_id = uuid.uuid4()
    files = (
        [
            ATSResumeFile(
                id=uuid.uuid4(),
                ats_resume_version_id=version_id,
                format=ATSFileFormat.PDF,
                s3_bucket="bucket",
                s3_key=f"key-{version_id}.pdf",
                file_size_bytes=100,
            )
        ]
        if completed
        else []
    )
    version = ATSResumeVersion(
        id=version_id,
        ats_resume_id=uuid.uuid4(),
        version_number=1,
        is_current=True,
        status=ATSVersionStatus.COMPLETED if completed else ATSVersionStatus.PENDING,
        source_resume_version_id=uuid.uuid4(),
        files=files,
    )
    resume = ATSResume(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        source_resume_id=uuid.uuid4(),
        target_job_id=target_job_id,
        title=title,
        company_name=company_name,
        versions=[version],
        created_at=_ts(minutes_ago),
        updated_at=_ts(minutes_ago),
    )
    return resume


def _plain_resume(*, minutes_ago: int = 0, parsed: bool = True) -> Resume:
    version = ResumeVersion(
        id=uuid.uuid4(),
        resume_id=uuid.uuid4(),
        version_number=1,
        is_current=True,
        status=ResumeVersionStatus.PARSED if parsed else ResumeVersionStatus.UPLOADED,
        content_type=ResumeContentType.PDF,
        original_filename="resume.pdf",
        s3_bucket="bucket",
        s3_key="plain-resume.pdf",
        file_size_bytes=100,
    )
    return Resume(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        title="My Resume",
        versions=[version],
        created_at=_ts(minutes_ago),
        updated_at=_ts(minutes_ago),
    )


def _job(*, job_id=None, company_name="Acme") -> Job:
    company = Company(id=uuid.uuid4(), name=company_name, normalized_name=company_name.lower())
    return Job(
        id=job_id or uuid.uuid4(),
        company_id=company.id,
        source="greenhouse",
        source_job_id="123",
        source_url="https://boards.greenhouse.io/acme/jobs/123",
        title="Software Engineer",
        description="desc",
        content_hash="hash",
        company=company,
    )


def test_prefers_exact_job_targeted_ats_resume() -> None:
    job = _job()
    generic = _ats_resume(minutes_ago=0, title="Generic")
    targeted = _ats_resume(target_job_id=job.id, minutes_ago=10, title="Targeted")

    selection = select_resume(ats_resumes=[generic, targeted], resumes=[], job=job)
    assert selection.display_name == "Targeted"
    assert selection.is_ats_optimized is True


def test_falls_back_to_company_name_match() -> None:
    job = _job(company_name="Acme")
    company_match = _ats_resume(company_name="Acme", minutes_ago=0, title="For Acme")
    other = _ats_resume(company_name="Other Co", minutes_ago=0, title="For Other")

    selection = select_resume(ats_resumes=[company_match, other], resumes=[], job=job)
    assert selection.display_name == "For Acme"


def test_falls_back_to_most_recent_completed_ats_resume() -> None:
    job = _job()
    older = _ats_resume(minutes_ago=20, title="Older")
    newer = _ats_resume(minutes_ago=1, title="Newer")

    selection = select_resume(ats_resumes=[older, newer], resumes=[], job=job)
    assert selection.display_name == "Newer"


def test_skips_incomplete_ats_resume_and_falls_back_to_plain_resume() -> None:
    job = _job()
    incomplete = _ats_resume(minutes_ago=0, completed=False)
    plain = _plain_resume()

    selection = select_resume(ats_resumes=[incomplete], resumes=[plain], job=job)
    assert selection.source == "resume"
    assert selection.is_ats_optimized is False


def test_raises_when_nothing_usable_is_available() -> None:
    with pytest.raises(ValidationError):
        select_resume(ats_resumes=[], resumes=[_plain_resume(parsed=False)], job=None)


def test_explicit_override_takes_priority() -> None:
    job = _job()
    preferred = _ats_resume(minutes_ago=50, title="Preferred")
    recent = _ats_resume(minutes_ago=0, title="Most Recent")

    selection = select_resume(
        ats_resumes=[preferred, recent], resumes=[], job=job, explicit_ats_resume_id=preferred.id
    )
    assert selection.display_name == "Preferred"
