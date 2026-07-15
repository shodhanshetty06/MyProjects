"""ATS Resume Generator module — Celery tasks.

Two tasks, chained and routed to two distinct queues (see
`app/workers/celery_app.py` `task_routes`):

    optimize_ats_resume_task        -> "ats_optimization" queue (Claude call)
    generate_ats_resume_files_task  -> "ats_files" queue (DOCX/PDF + S3 upload)

Sync-Celery-wrapping-`asyncio.run` convention, same as
`resume_parser/tasks.py`: each task is a thin wrapper so the existing async
DB session (`AsyncSessionLocal`) can be reused without a parallel sync
SQLAlchemy stack.
"""
from __future__ import annotations

import uuid
from asyncio import run as run_async

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal
from app.integrations.s3_client import S3Storage, get_s3_client
from app.modules.ats_resume_generator import documents, scoring
from app.modules.ats_resume_generator.models import ATSFileFormat, ATSVersionStatus
from app.modules.ats_resume_generator.repository import ATSResumeVersionRepository
from app.modules.ats_resume_generator.schemas import OptimizedContent
from app.services.ai.service import AIService
from app.workers.base_task import BaseTask
from app.workers.celery_app import celery_app

logger = get_logger(__name__)

OPTIMIZE_SYSTEM_PROMPT = (
    "You are an ATS resume optimization assistant. You are given a target job description and "
    "a candidate's existing resume content (current summary, skills, projects, and the full "
    "original resume text for grounding). Rewrite ONLY the summary, skills, and projects to "
    "better align with the job description and surface relevant keywords.\n\n"
    "Hard rules — follow these exactly:\n"
    "1. Never invent, exaggerate, or infer experience, employers, job titles, dates, companies, "
    "or outcomes that are not clearly present in the provided resume content.\n"
    "2. Every skill you output must already be evidenced in the original resume content (the "
    "skills list, project descriptions, or full resume text) — do not add a skill just because "
    "it appears in the job description if the candidate hasn't demonstrated it.\n"
    "3. Do not add new projects, remove existing projects, or change a project's technologies "
    "or outcomes beyond what the original text states — only improve phrasing, clarity, and "
    "keyword alignment of what's already there.\n"
    "4. Separately list important keywords (skills/technologies/terms) mentioned in the job "
    "description — these are for scoring only and do not need to all appear in the rewritten text."
)


def _build_optimize_prompt(
    *,
    job_description: str,
    original_summary: str,
    original_skills: list[str],
    original_projects: list[dict],
    raw_resume_text: str,
) -> str:
    project_lines = []
    for project in original_projects:
        technologies = ", ".join(project.get("technologies") or [])
        highlights = "; ".join(project.get("highlights") or [])
        project_lines.append(
            f"- {project.get('name', '')}: {project.get('description') or ''} "
            f"[technologies: {technologies}] [highlights: {highlights}]"
        )

    return (
        f"JOB DESCRIPTION:\n{job_description}\n\n"
        f"CURRENT SUMMARY:\n{original_summary}\n\n"
        f"CURRENT SKILLS:\n{', '.join(original_skills)}\n\n"
        f"CURRENT PROJECTS:\n" + ("\n".join(project_lines) or "(none)") + "\n\n"
        f"FULL ORIGINAL RESUME TEXT (for grounding only — do not copy sections outside "
        f"summary/skills/projects from here):\n{raw_resume_text}"
    )


@celery_app.task(base=BaseTask, bind=True, name="ats_resume_generator.optimize")
def optimize_ats_resume_task(self, version_id: str) -> None:  # noqa: ANN001 -- Celery `self`
    run_async(_optimize_ats_resume(uuid.UUID(version_id)))


@celery_app.task(base=BaseTask, bind=True, name="ats_resume_generator.generate_files")
def generate_ats_resume_files_task(self, version_id: str) -> None:  # noqa: ANN001
    run_async(_generate_ats_resume_files(uuid.UUID(version_id)))


async def _optimize_ats_resume(version_id: uuid.UUID) -> None:
    settings = get_settings()
    succeeded = False

    async with AsyncSessionLocal() as session:
        repo = ATSResumeVersionRepository(session)
        version = await repo.get_by_id(version_id=version_id)
        if version is None:
            logger.warning(
                "ATS resume version not found for optimization",
                extra={"extra_fields": {"version_id": str(version_id)}},
            )
            return

        await repo.update(version, status=ATSVersionStatus.OPTIMIZING)
        await session.commit()

        try:
            source = await repo.get_source_resume_version(
                resume_version_id=version.source_resume_version_id
            )
            raw_text = source.parsed_data.raw_text if source and source.parsed_data else ""

            prompt = _build_optimize_prompt(
                job_description=version.job_description,
                original_summary=version.original_summary,
                original_skills=version.original_skills,
                original_projects=version.original_projects,
                raw_resume_text=raw_text,
            )

            ai_service = AIService()
            extracted = await ai_service.extract_structured(
                text=prompt,
                schema=OptimizedContent.model_json_schema(),
                system=OPTIMIZE_SYSTEM_PROMPT,
            )
            optimized = OptimizedContent.model_validate(extracted)
            optimized_projects = [item.model_dump() for item in optimized.projects]

            resume_text = scoring.build_resume_search_text(
                summary=optimized.summary,
                skills=optimized.skills,
                projects=optimized_projects,
                experience=version.experience,
                education=version.education,
                certifications=version.certifications,
            )
            keyword_result = scoring.keyword_coverage(optimized.keywords, resume_text)

            bullets: list[str] = []
            for job in version.experience:
                bullets.extend(job.get("highlights") or [])
            for project in optimized_projects:
                bullets.extend(project.get("highlights") or [])

            structure_result = scoring.structure_score(
                has_summary=bool(optimized.summary),
                has_skills=bool(optimized.skills),
                has_experience=bool(version.experience),
                has_education=bool(version.education),
                has_projects=bool(optimized_projects),
                bullet_highlights=bullets,
            )
            ats_score = scoring.compute_ats_score(
                keyword_score=keyword_result.score,
                structure_score_value=structure_result.score,
                keyword_weight=settings.ATS_KEYWORD_SCORE_WEIGHT,
                structure_weight=settings.ATS_STRUCTURE_SCORE_WEIGHT,
            )

            await repo.update(
                version,
                status=ATSVersionStatus.GENERATING_FILES,
                optimized_summary=optimized.summary,
                optimized_skills=optimized.skills,
                optimized_projects=optimized_projects,
                keywords=optimized.keywords,
                matched_keywords=keyword_result.matched,
                missing_keywords=keyword_result.missing,
                keyword_score=round(keyword_result.score, 2),
                ats_score=round(ats_score, 2),
                score_breakdown={
                    "keyword_score": round(keyword_result.score, 2),
                    "structure_score": round(structure_result.score, 2),
                    "structure_checks": structure_result.checks,
                },
                model_used=settings.ANTHROPIC_MODEL,
                error_message=None,
            )
            await session.commit()
            succeeded = True
        except Exception as exc:
            await session.rollback()
            await repo.update(version, status=ATSVersionStatus.FAILED, error_message=str(exc))
            await session.commit()
            logger.error(
                "ATS resume optimization failed",
                extra={"extra_fields": {"version_id": str(version_id), "error": str(exc)}},
            )

    if succeeded:
        generate_ats_resume_files_task.delay(str(version_id))


def _s3_key(user_id: uuid.UUID, ats_resume_id: uuid.UUID, version_id: uuid.UUID, fmt: ATSFileFormat) -> str:
    extension = "docx" if fmt == ATSFileFormat.DOCX else "pdf"
    return f"ats-resumes/{user_id}/{ats_resume_id}/{version_id}/resume.{extension}"


async def _generate_ats_resume_files(version_id: uuid.UUID) -> None:
    async with AsyncSessionLocal() as session:
        repo = ATSResumeVersionRepository(session)
        version = await repo.get_by_id(version_id=version_id)
        if version is None or version.ats_resume is None:
            logger.warning(
                "ATS resume version not found for file generation",
                extra={"extra_fields": {"version_id": str(version_id)}},
            )
            return

        try:
            content = documents.ATSDocumentContent(
                title=version.ats_resume.title,
                summary=version.optimized_summary,
                skills=version.optimized_skills,
                experience=version.experience,
                projects=version.optimized_projects,
                education=version.education,
                certifications=version.certifications,
                languages=version.languages,
                achievements=version.achievements,
            )
            docx_bytes = documents.generate_docx(content)
            pdf_bytes = documents.generate_pdf(content)

            s3_storage = S3Storage(client=get_s3_client())
            user_id = version.ats_resume.user_id
            ats_resume_id = version.ats_resume_id

            docx_key = _s3_key(user_id, ats_resume_id, version.id, ATSFileFormat.DOCX)
            pdf_key = _s3_key(user_id, ats_resume_id, version.id, ATSFileFormat.PDF)
            s3_storage.upload_bytes(docx_bytes, docx_key)
            s3_storage.upload_bytes(pdf_bytes, pdf_key)

            await repo.add_file(
                ats_resume_version_id=version.id,
                format=ATSFileFormat.DOCX,
                s3_bucket=s3_storage.bucket,
                s3_key=docx_key,
                file_size_bytes=len(docx_bytes),
            )
            await repo.add_file(
                ats_resume_version_id=version.id,
                format=ATSFileFormat.PDF,
                s3_bucket=s3_storage.bucket,
                s3_key=pdf_key,
                file_size_bytes=len(pdf_bytes),
            )
            await repo.update(version, status=ATSVersionStatus.COMPLETED, error_message=None)
            await session.commit()
        except Exception as exc:
            await session.rollback()
            await repo.update(version, status=ATSVersionStatus.FAILED, error_message=str(exc))
            await session.commit()
            logger.error(
                "ATS resume file generation failed",
                extra={"extra_fields": {"version_id": str(version_id), "error": str(exc)}},
            )
            return

    logger.info(
        "ATS resume files generated",
        extra={"extra_fields": {"version_id": str(version_id)}},
    )
