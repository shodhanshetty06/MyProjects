"""Unit test for `matching/tasks.py::_match_new_job_for_active_users`'s
realtime hook — asserts that, alongside the existing notification, it also
publishes a `new_matching_jobs` event on the `search:updates:{user_id}`
channel (see `app.modules.search.realtime`), the "refresh search" leg
added on top of the existing match+notify pipeline (plan section 7).

Needs a reachable Postgres (job_ingestion + resume_parser + matching tables
migrated):

    docker compose exec api alembic upgrade head
    docker compose exec api pytest tests/modules/matching/test_tasks.py

`NotificationService.notify` and `search.realtime.publish_search_event` are
monkeypatched at the function level (same approach as
`matching/test_router.py`'s `_fake_external_dependencies` fixture), so no
running Redis/SMTP/worker is required.
"""

from __future__ import annotations

import uuid

import pytest

from app.core.config import get_settings
from app.db.session import AsyncSessionLocal
from app.modules.job_ingestion.models import (
    ATSPlatform,
    Company,
    EmploymentType,
    Job,
    JobEmbedding,
    JobStatus,
    WorkMode,
)
from app.modules.matching.models import MatchPreference
from app.modules.matching import tasks as matching_tasks
from app.modules.notification.service import NotificationService
from app.modules.resume_parser.models import (
    Resume,
    ResumeContentType,
    ResumeEmbedding,
    ResumeParsedData,
    ResumeVersion,
    ResumeVersionStatus,
)
from app.modules.search import realtime as search_realtime

VECTOR_DIMENSIONS = get_settings().VECTOR_DIMENSIONS


def _embedding(seed: float) -> list[float]:
    return [seed] * VECTOR_DIMENSIONS


async def _seed_job(*, embedding: list[float]) -> Job:
    async with AsyncSessionLocal() as session:
        company = Company(name="Acme", normalized_name=f"acme-{uuid.uuid4()}")
        session.add(company)
        await session.flush()

        job = Job(
            company_id=company.id,
            source=ATSPlatform.CUSTOM_HTML,
            source_job_id=str(uuid.uuid4()),
            source_url="https://example.com/job",
            title="Senior Backend Engineer",
            description="Build backend services.",
            employment_type=EmploymentType.FULL_TIME,
            work_mode=WorkMode.REMOTE,
            status=JobStatus.ACTIVE,
            content_hash=str(uuid.uuid4()),
        )
        session.add(job)
        await session.flush()
        session.add(JobEmbedding(job_id=job.id, embedding=embedding, model_used="test-model"))
        await session.commit()
        await session.refresh(job)
        return job


async def _seed_active_user(*, embedding: list[float]) -> uuid.UUID:
    user_id = uuid.uuid4()
    async with AsyncSessionLocal() as session:
        session.add(
            MatchPreference(
                user_id=user_id,
                preferred_countries=[],
                preferred_work_modes=[],
                preferred_employment_types=[],
                is_active=True,
            )
        )
        resume = Resume(user_id=user_id, title="My Resume")
        session.add(resume)
        await session.flush()

        version = ResumeVersion(
            resume_id=resume.id,
            version_number=1,
            is_current=True,
            status=ResumeVersionStatus.PARSED,
            content_type=ResumeContentType.PDF,
            original_filename="resume.pdf",
            s3_bucket="test-bucket",
            s3_key=f"resumes/{uuid.uuid4()}.pdf",
            file_size_bytes=1024,
        )
        session.add(version)
        await session.flush()
        session.add(
            ResumeParsedData(
                resume_version_id=version.id,
                summary="Backend engineer.",
                skills=["Python"],
                model_used="test-model",
            )
        )
        session.add(
            ResumeEmbedding(
                resume_version_id=version.id, embedding=embedding, model_used="test-model"
            )
        )
        await session.commit()
    return user_id


async def test_match_new_job_publishes_search_realtime_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = get_settings()
    # Force the notify threshold low so this test's exact-match embedding
    # (score 1.0 similarity) always clears it, regardless of the
    # environment's configured default.
    monkeypatch.setattr(settings, "MATCHING_NEW_JOB_NOTIFY_THRESHOLD", 0.0)

    async def _fake_notify(self, **kwargs) -> None:  # noqa: ANN001
        return None

    monkeypatch.setattr(NotificationService, "notify", _fake_notify)

    published: list[tuple] = []

    async def _fake_publish(user_id, event) -> None:  # noqa: ANN001
        published.append((user_id, event))

    monkeypatch.setattr(search_realtime, "publish_search_event", _fake_publish)

    embedding = _embedding(0.5)
    job = await _seed_job(embedding=embedding)
    user_id = await _seed_active_user(embedding=embedding)

    await matching_tasks._match_new_job_for_active_users(job.id)

    # The dev DB is shared/persistent across test runs, so other
    # `MatchPreference(is_active=True)` rows seeded by earlier test
    # sessions may also get matched+published here — assert this test's
    # own user shows up with the expected event shape, not that it's the
    # only one.
    matching_events_for_user = [
        event
        for uid, event in published
        if uid == user_id and event.get("type") == "new_matching_jobs"
    ]
    assert len(matching_events_for_user) == 1
    event = matching_events_for_user[0]
    assert event["job_id"] == str(job.id)
    assert "match_score" in event
