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
from asyncio import to_thread
from io import BytesIO

from pypdf import PdfReader
from sqlalchemy.exc import OperationalError

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal
from app.dependency_injection.container import get_container
from app.modules.ats_resume_generator import scoring
from app.modules.ats_resume_generator.content import build_resume_content
from app.modules.ats_resume_generator.docx_renderer import render_docx
from app.modules.ats_resume_generator.models import ATSFileFormat, ATSVersionStatus
from app.modules.ats_resume_generator.pdf_converter import convert_docx_to_pdf
from app.modules.ats_resume_generator.repository import ATSResumeVersionRepository
from app.modules.ats_resume_generator.schemas import OptimizedContent
from app.modules.ats_resume_generator.templates import get_template
from app.modules.ats_resume_generator.validation import check_page_overflow, validate_resume_content
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
    "description — these are for scoring only and do not need to all appear in the rewritten text.\n"
    "5. `grouped_skills` must be a strict repartition of `skills` — every skill in `skills` "
    "appears in exactly one category of `grouped_skills`, and no category may contain a skill "
    "that isn't in `skills`. Choose categories for clarity (e.g. 'Languages', 'Frameworks & "
    "Libraries', 'Tools & Platforms', 'Databases'), not exhaustiveness — a handful of clear "
    "groups beats many overlapping ones."
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


async def _mark_failed_best_effort(version_id: uuid.UUID, error_message: str) -> None:
    """Last-resort write when a task exhausts its DB-connection retries — a
    fresh session/pool checkout, since whatever session the failed attempt
    was using may itself be the thing that couldn't get a connection. Without
    this, a version that never got a session at all (see `_generate_ats_resume_files`'s
    initial fetch, which sits before its own try/except) stays stuck at
    `optimizing`/`generating_files` forever: `POLL_STATUSES` on the frontend
    (`GenerateAtsResumeDialog.tsx`, `ATSResumeGeneratorPage.tsx`) never stops
    polling, so the user just watches a spinner with no way to know it's dead.
    """
    try:
        async with AsyncSessionLocal() as session:
            repo = ATSResumeVersionRepository(session)
            version = await repo.get_by_id(version_id=version_id)
            if version is not None and version.status != ATSVersionStatus.COMPLETED:
                await repo.update(version, status=ATSVersionStatus.FAILED, error_message=error_message)
                await session.commit()
    except Exception:
        logger.error(
            "Could not mark ATS resume version failed after retry exhaustion",
            extra={"extra_fields": {"version_id": str(version_id)}},
        )


@celery_app.task(
    base=BaseTask,
    bind=True,
    name="ats_resume_generator.optimize",
    autoretry_for=(OperationalError,),
    retry_backoff=True,
    retry_backoff_max=30,
    max_retries=5,
)
def optimize_ats_resume_task(self, version_id: str) -> None:  # noqa: ANN001 -- Celery `self`
    try:
        run_async(_optimize_ats_resume(uuid.UUID(version_id)))
    except OperationalError:
        if self.request.retries >= self.max_retries:
            run_async(_mark_failed_best_effort(uuid.UUID(version_id), "Database unavailable — please try again."))
        raise


@celery_app.task(
    base=BaseTask,
    bind=True,
    name="ats_resume_generator.generate_files",
    autoretry_for=(OperationalError,),
    retry_backoff=True,
    retry_backoff_max=30,
    max_retries=5,
)
def generate_ats_resume_files_task(self, version_id: str) -> None:  # noqa: ANN001
    try:
        run_async(_generate_ats_resume_files(uuid.UUID(version_id)))
    except OperationalError:
        if self.request.retries >= self.max_retries:
            run_async(_mark_failed_best_effort(uuid.UUID(version_id), "Database unavailable — please try again."))
        raise


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

            ai_service = get_container().resolve("ai_service")
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
                grouped_skills=[item.model_dump() for item in optimized.grouped_skills],
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
                model_used=ai_service.model_name,
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
            content = build_resume_content(version)
            validation_result = validate_resume_content(content)
            if not validation_result.passed:
                await repo.update(
                    version,
                    status=ATSVersionStatus.FAILED,
                    error_message="; ".join(validation_result.errors),
                    validation_warnings=validation_result.warnings,
                )
                await session.commit()
                logger.warning(
                    "ATS resume file generation blocked by validation",
                    extra={
                        "extra_fields": {
                            "version_id": str(version_id),
                            "errors": validation_result.errors,
                        }
                    },
                )
                return

            template = get_template(version.template_id)
            docx_bytes = render_docx(content, template)
            pdf_bytes = await to_thread(convert_docx_to_pdf, docx_bytes)

            warnings = list(validation_result.warnings)
            overflow_warning = check_page_overflow(len(PdfReader(BytesIO(pdf_bytes)).pages))
            if overflow_warning:
                warnings.append(overflow_warning)

            storage_provider = get_container().resolve("storage_provider")
            user_id = version.ats_resume.user_id
            ats_resume_id = version.ats_resume_id

            docx_key = _s3_key(user_id, ats_resume_id, version.id, ATSFileFormat.DOCX)
            pdf_key = _s3_key(user_id, ats_resume_id, version.id, ATSFileFormat.PDF)
            storage_provider.upload_bytes(docx_bytes, docx_key)
            storage_provider.upload_bytes(pdf_bytes, pdf_key)

            await repo.add_file(
                ats_resume_version_id=version.id,
                format=ATSFileFormat.DOCX,
                s3_bucket=storage_provider.bucket,
                s3_key=docx_key,
                file_size_bytes=len(docx_bytes),
            )
            await repo.add_file(
                ats_resume_version_id=version.id,
                format=ATSFileFormat.PDF,
                s3_bucket=storage_provider.bucket,
                s3_key=pdf_key,
                file_size_bytes=len(pdf_bytes),
            )
            await repo.update(
                version,
                status=ATSVersionStatus.COMPLETED,
                error_message=None,
                validation_warnings=warnings,
            )
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

    # Smart Trigger: ATS Resume Generated -> update workspace. The version is
    # already stored (above) and Auto Apply already prefers a COMPLETED ATS
    # version over a plain resume (`auto_apply.resume_selection.select_resume`)
    # — no further plumbing needed there.
    from app.modules.workspace import realtime as workspace_realtime

    await workspace_realtime.publish_workspace_event(
        user_id,
        {
            "type": "ats_resume_ready",
            "ats_resume_version_id": str(version_id),
        },
    )
