"""Resume Parser module — Celery tasks.

Two tasks, routed to two distinct queues (see `app/workers/celery_app.py`
`task_routes` and the `worker-parsing` / `worker-embedding` services in
`docker-compose.yml`):

    parse_resume_task              -> "parsing" queue
    generate_resume_embedding_task -> "embedding" queue

Celery workers are synchronous processes; each task is a thin sync wrapper
around an `asyncio.run(...)` call so the existing async DB session
(`AsyncSessionLocal`) can be reused without a parallel sync SQLAlchemy stack.
"""
from __future__ import annotations

import uuid
from asyncio import run as run_async

from sqlalchemy import select

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal
from app.integrations.s3_client import S3Storage, get_s3_client
from app.modules.resume_parser.extraction import extract_text
from app.modules.resume_parser.models import ResumeEmbedding, ResumeParsedData, ResumeVersionStatus
from app.modules.resume_parser.repository import ResumeVersionRepository
from app.modules.resume_parser.schemas import ParsedResumeData
from app.services.ai.service import AIService
from app.workers.base_task import BaseTask
from app.workers.celery_app import celery_app

logger = get_logger(__name__)

EXTRACTION_SYSTEM_PROMPT = (
    "You are a resume parsing assistant. Extract the requested fields from the resume text "
    "exactly as written on the resume. Do not invent information that is not present in the "
    "text. If a section is absent, return an empty list (or empty string for the summary) "
    "for it rather than guessing."
)


@celery_app.task(base=BaseTask, bind=True, name="resume_parser.parse_resume")
def parse_resume_task(self, resume_version_id: str) -> None:  # noqa: ANN001 -- Celery `self`
    run_async(_parse_resume(uuid.UUID(resume_version_id)))


@celery_app.task(base=BaseTask, bind=True, name="resume_parser.generate_embedding")
def generate_resume_embedding_task(self, resume_version_id: str) -> None:  # noqa: ANN001
    run_async(_generate_embedding(uuid.UUID(resume_version_id)))


async def _parse_resume(version_id: uuid.UUID) -> None:
    settings = get_settings()

    async with AsyncSessionLocal() as session:
        repo = ResumeVersionRepository(session)
        version = await repo.get(version_id)
        if version is None:
            logger.warning(
                "Resume version not found for parsing",
                extra={"extra_fields": {"version_id": str(version_id)}},
            )
            return

        await repo.update(version, status=ResumeVersionStatus.PARSING)
        await session.commit()

        try:
            s3_storage = S3Storage(client=get_s3_client())
            raw_bytes = s3_storage.download_bytes(version.s3_key)
            text = extract_text(version.content_type, raw_bytes)

            ai_service = AIService()
            extracted = await ai_service.extract_structured(
                text=text,
                schema=ParsedResumeData.model_json_schema(),
                system=EXTRACTION_SYSTEM_PROMPT,
            )
            parsed = ParsedResumeData.model_validate(extracted)

            session.add(
                ResumeParsedData(
                    resume_version_id=version.id,
                    summary=parsed.summary,
                    skills=parsed.skills,
                    experience=[item.model_dump() for item in parsed.experience],
                    education=[item.model_dump() for item in parsed.education],
                    projects=[item.model_dump() for item in parsed.projects],
                    certifications=[item.model_dump() for item in parsed.certifications],
                    achievements=parsed.achievements,
                    languages=[item.model_dump() for item in parsed.languages],
                    raw_text=text,
                    model_used=settings.ANTHROPIC_MODEL,
                )
            )
            await repo.update(version, status=ResumeVersionStatus.PARSED, error_message=None)
            await session.commit()
        except Exception as exc:
            await session.rollback()
            await repo.update(version, status=ResumeVersionStatus.FAILED, error_message=str(exc))
            await session.commit()
            logger.error(
                "Resume parsing failed",
                extra={"extra_fields": {"version_id": str(version_id), "error": str(exc)}},
            )
            return

    generate_resume_embedding_task.delay(str(version_id))


def _embedding_input(parsed: ResumeParsedData) -> str:
    sections = [
        parsed.summary,
        "Skills: " + ", ".join(parsed.skills) if parsed.skills else "",
        "Experience: " + "; ".join(
            f"{exp.get('title', '')} at {exp.get('company', '')} — {exp.get('description', '') or ''}"
            for exp in parsed.experience
        ),
        "Education: " + "; ".join(
            f"{edu.get('degree', '')} in {edu.get('field_of_study', '')} from {edu.get('institution', '')}"
            for edu in parsed.education
        ),
    ]
    return "\n".join(s for s in sections if s)


async def _generate_embedding(version_id: uuid.UUID) -> None:
    settings = get_settings()

    async with AsyncSessionLocal() as session:
        parsed = (
            await session.execute(
                select(ResumeParsedData).where(ResumeParsedData.resume_version_id == version_id)
            )
        ).scalars().one_or_none()
        if parsed is None:
            logger.warning(
                "No parsed data found for embedding generation",
                extra={"extra_fields": {"version_id": str(version_id)}},
            )
            return

        ai_service = AIService()
        vector = await ai_service.generate_embedding(text=_embedding_input(parsed))

        session.add(
            ResumeEmbedding(
                resume_version_id=version_id,
                embedding=vector,
                model_used=settings.VOYAGE_MODEL,
            )
        )
        await session.commit()

    logger.info(
        "Resume embedding generated",
        extra={"extra_fields": {"version_id": str(version_id)}},
    )

    # Chain into the matching engine now that this version has both parsed
    # data and an embedding — the same kind of task-chaining as
    # `parse_resume_task` -> `generate_resume_embedding_task` above.
    from app.modules.matching.tasks import compute_resume_matches_task

    compute_resume_matches_task.delay(str(version_id))
