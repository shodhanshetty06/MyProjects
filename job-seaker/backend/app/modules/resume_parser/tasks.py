"""Resume Parser module — Celery tasks.

These are the first four links of the auto-orchestrated "ResumeUploaded"
pipeline (see `app.modules.resume_parser.repository.PIPELINE_STEPS` for the
full chain and progress weights):

    parse_resume_task              -> "parsing" queue
    generate_resume_embedding_task -> "embedding" queue
    calculate_resume_score_task    -> "embedding" queue
    run_ats_baseline_analysis_task -> "embedding" queue

...then hands off to `matching.tasks.compute_resume_matches_task`. Every
task threads `user_id` through as an explicit argument (rather than each
task re-deriving it from `ResumeVersion.resume.user_id`) so downstream tasks
don't need an extra join just to publish a workspace progress event.

Celery workers are synchronous processes; each task is a thin sync wrapper
around an `asyncio.run(...)` call so the existing async DB session
(`AsyncSessionLocal`) can be reused without a parallel sync SQLAlchemy stack.
"""
from __future__ import annotations

import uuid
from asyncio import gather
from asyncio import run as run_async
from copy import deepcopy

from sqlalchemy import func, or_, select

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal
from app.dependency_injection.container import get_container
from app.modules.resume_parser.extraction import extract_text
from app.modules.resume_parser.models import ResumeEmbedding, ResumeParsedData, ResumeVersionStatus
from app.modules.resume_parser.repository import (
    ResumePipelineRunRepository,
    ResumeVersionRepository,
)
from app.modules.resume_parser.schemas import (
    ParsedResumeData,
    ParsedResumeDataNarrative,
    ParsedResumeDataStructured,
)
from app.workers.base_task import BaseTask
from app.workers.celery_app import celery_app

logger = get_logger(__name__)

EXTRACTION_SYSTEM_PROMPT = (
    "You are a resume parsing assistant. Extract the requested fields from the resume text "
    "exactly as written on the resume. Do not invent information that is not present in the "
    "text. If a section is truly absent from the resume, return an empty list for it rather "
    "than guessing — but read the ENTIRE document first, since resumes vary in layout and "
    "section order and a section may use a different heading than you expect (e.g. 'Profile' "
    "or 'Objective' for the summary, 'Technical Skills' or 'Core Competencies' for skills, "
    "'Employment History' or 'Internship' for experience, 'Academic Projects' for projects, "
    "'Licenses & Certifications' or 'Courses and Workshops' for certifications). The `summary` "
    "field is the one exception to 'leave it empty when absent': never return an empty string "
    "for `summary` — if the resume has no explicit summary/objective/profile section, "
    "synthesize a concise 2-4 sentence one yourself from the person's experience and skills. "
    "Never leave a field empty just because its exact heading name doesn't "
    "match — match by meaning, not literal heading text. Similarly, within one entry, never "
    "drop the whole entry just because one sub-field (e.g. a job title) isn't explicitly "
    "labeled — use the closest available label (e.g. a department or project name standing in "
    "for a job title) rather than omitting the entry.\n\n"
    "Pay special attention to the difference between `skills` and `languages`: programming, "
    "scripting, query, and markup languages (e.g. C, C++, C#, Java, Python, JavaScript, SQL, "
    "HTML) always belong in `skills`, even if the resume lists them under a heading literally "
    "called 'Languages'. The `languages` field is ONLY for spoken/written human languages "
    "(e.g. English, Hindi, Spanish) — never put a programming language there.\n\n"
    "For every project entry, always copy its explanatory text into `description` — this is "
    "the paragraph or bullet points written under the project title/technologies line, usually "
    "starting with something like 'The aim of...' or describing what the project does or what "
    "was built. Do not leave `description` null just because it takes multiple sentences to "
    "copy — extracting it is just as mandatory as extracting the project `name`.\n\n"
    "For `contact`: the name/contact block is almost always the first few lines of the "
    "document. Extract the full name, phone, email, LinkedIn/GitHub/portfolio URLs, and "
    "location exactly as written, whether or not they're explicitly labeled (e.g. a bare "
    "'linkedin.com/in/...' URL with no 'LinkedIn:' label still belongs in `linkedin_url`). "
    "Never fabricate a contact field that isn't present — leave it null instead."
)

_BASELINE_SKILL_LIMIT = 15


def _force_required_string(schema: dict, def_name: str, field: str) -> dict:
    """Return a copy of `schema` with `def_name.field` marked as a required,
    non-nullable string.

    Ollama's constrained decoding (the `format=schema` grammar) treats a field
    that is optional-and-nullable as skippable, so the default `qwen2.5:3b`
    model routinely omits `ProjectItem.description` — leaving the "Description"
    box empty in the UI — even though `EXTRACTION_SYSTEM_PROMPT` tells it to
    always copy the project's explanatory text. Collapsing the field's
    `anyOf: [string, null]` (Pydantic's `str | None`) down to a plain required
    string makes the grammar refuse to close the object without emitting a
    string, so the description is actually extracted. We tweak only the schema
    handed to the extractor — the Pydantic models stay tolerant so older parsed
    rows that legitimately lack a description still validate on read.
    """
    schema = deepcopy(schema)
    target = schema.get("$defs", {}).get(def_name)
    if not target:
        return schema
    prop = target.get("properties", {}).get(field)
    if prop is not None:
        prop.pop("anyOf", None)
        prop.pop("default", None)
        prop["type"] = "string"
        required = target.setdefault("required", [])
        if field not in required:
            required.append(field)
    return schema


def _force_min_length(schema: dict, field: str, min_length: int, def_name: str | None = None) -> dict:
    """Return a copy of `schema` with `field` given a `minLength` constraint,
    so the model's constrained-decoding grammar can't satisfy the "required
    string" rule by emitting `""`. Targets a top-level field by default, or
    `$defs[def_name].properties[field]` when `def_name` is given (e.g.
    `ContactInfo.full_name`, nested under the `contact` property).

    `summary` and `contact.full_name` are both declared as plain required
    `str` (not `str | None`), so neither suffers from the `anyOf`-skippable
    defect `_force_required_string` fixes for `projects.description` — but a
    required-string grammar is still satisfiable with an empty string. Pinning
    `minLength` here closes that loophole at the schema level, independent of
    how forcefully `EXTRACTION_SYSTEM_PROMPT` says "never leave this blank"
    (observed: the default local model ignores that instruction for
    `full_name` on some resumes despite the name being plainly on the page).
    """
    schema = deepcopy(schema)
    target = schema.get("$defs", {}).get(def_name) if def_name else schema
    if target is None:
        return schema
    prop = target.get("properties", {}).get(field)
    if prop is not None:
        prop["minLength"] = min_length
        # A field with a `default` (e.g. `full_name: str = Field(default="")`)
        # isn't in Pydantic's generated `required` list, so Ollama's grammar
        # can satisfy the object by omitting it entirely — falling back to
        # the empty-string default without ever consulting `minLength`.
        # Dropping the default and marking it required forces the grammar to
        # actually emit a non-empty string.
        prop.pop("default", None)
        required = target.setdefault("required", [])
        if field not in required:
            required.append(field)
    return schema


def _drop_blank(values: list[str]) -> list[str]:
    return [v for v in values if v and v.strip()]


async def _advance_pipeline(
    *, user_id: uuid.UUID, resume_version_id: uuid.UUID, step: str, status: str, error: str | None = None
) -> None:
    """Update the durable `ResumePipelineRun` row for one step transition and
    push a live event over the workspace WebSocket channel. Best-effort: a
    missing run (e.g. bookkeeping race) never blocks the actual pipeline
    work, it's just logged.
    """
    from app.modules.workspace import realtime as workspace_realtime

    async with AsyncSessionLocal() as session:
        repo = ResumePipelineRunRepository(session)
        run = await repo.advance(
            resume_version_id=resume_version_id, step=step, status=status, error=error
        )
        await session.commit()

    if run is None:
        logger.warning(
            "No pipeline run found to advance",
            extra={"extra_fields": {"resume_version_id": str(resume_version_id), "step": step}},
        )
        return

    await workspace_realtime.publish_workspace_event(
        user_id,
        {
            "type": "pipeline_progress",
            "run_id": str(run.id),
            "resume_version_id": str(resume_version_id),
            "step": step,
            "step_status": status,
            "status": run.status.value,
            "progress_pct": run.progress_pct,
        },
    )


@celery_app.task(base=BaseTask, bind=True, name="resume_parser.parse_resume")
def parse_resume_task(self, resume_version_id: str, user_id: str) -> None:  # noqa: ANN001 -- Celery `self`
    run_async(_parse_resume(uuid.UUID(resume_version_id), uuid.UUID(user_id)))


@celery_app.task(base=BaseTask, bind=True, name="resume_parser.generate_embedding")
def generate_resume_embedding_task(self, resume_version_id: str, user_id: str) -> None:  # noqa: ANN001
    run_async(_generate_embedding(uuid.UUID(resume_version_id), uuid.UUID(user_id)))


@celery_app.task(
    base=BaseTask,
    bind=True,
    name="resume_parser.calculate_score",
    autoretry_for=(Exception,),
    max_retries=3,
    retry_backoff=True,
)
def calculate_resume_score_task(self, resume_version_id: str, user_id: str) -> None:  # noqa: ANN001
    run_async(_calculate_resume_score(uuid.UUID(resume_version_id), uuid.UUID(user_id)))


@celery_app.task(
    base=BaseTask,
    bind=True,
    name="resume_parser.run_ats_baseline_analysis",
    autoretry_for=(Exception,),
    max_retries=3,
    retry_backoff=True,
)
def run_ats_baseline_analysis_task(self, resume_version_id: str, user_id: str) -> None:  # noqa: ANN001
    run_async(_run_ats_baseline_analysis(uuid.UUID(resume_version_id), uuid.UUID(user_id)))


async def _parse_resume(version_id: uuid.UUID, user_id: uuid.UUID) -> None:
    try:
        async with AsyncSessionLocal() as session:
            pipeline_repo = ResumePipelineRunRepository(session)
            await pipeline_repo.start(user_id=user_id, resume_version_id=version_id)
            await session.commit()
        await _advance_pipeline(
            user_id=user_id, resume_version_id=version_id, step="parsing", status="running"
        )
    except Exception as exc:
        # Anything unexpected here (e.g. a stale connection slipping past the
        # task_postrun cleanup) must still mark the version FAILED — otherwise
        # it's stuck at UPLOADED forever with nothing to break the frontend's
        # poll loop out of "Reading your resume...".
        logger.error(
            "Resume parsing failed to start",
            extra={"extra_fields": {"version_id": str(version_id), "error": str(exc)}},
        )
        async with AsyncSessionLocal() as session:
            repo = ResumeVersionRepository(session)
            version = await repo.get(version_id)
            if version is not None:
                await repo.update(version, status=ResumeVersionStatus.FAILED, error_message=str(exc)[:2000])
                await session.commit()
        return

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
            container = get_container()
            storage_provider = container.resolve("storage_provider")
            raw_bytes = storage_provider.download_bytes(version.s3_key)
            text = extract_text(version.content_type, raw_bytes)

            ai_service = container.resolve("ai_service")
            # Two calls instead of one: a single request against the full
            # 8-field/5-nested-type ParsedResumeData schema reliably fills
            # 1-2 fields and leaves the rest at their empty defaults on
            # smaller local models (the default `AI_PROVIDER=ollama` setup) —
            # trivially schema-valid, but useless. Splitting narrative fields
            # (summary/skills/achievements/languages) from structured list
            # fields (experience/education/projects/certifications) measurably
            # fixes this, confirmed against this project's default model.
            structured_schema = _force_required_string(
                ParsedResumeDataStructured.model_json_schema(), "ProjectItem", "description"
            )
            narrative_schema = _force_min_length(
                ParsedResumeDataNarrative.model_json_schema(), "summary", 1
            )
            narrative_schema = _force_min_length(
                narrative_schema, "full_name", 1, def_name="ContactInfo"
            )
            narrative_raw, structured_raw = await gather(
                ai_service.extract_structured(
                    text=text,
                    schema=narrative_schema,
                    system=EXTRACTION_SYSTEM_PROMPT,
                ),
                ai_service.extract_structured(
                    text=text,
                    schema=structured_schema,
                    system=EXTRACTION_SYSTEM_PROMPT,
                ),
            )
            narrative = ParsedResumeDataNarrative.model_validate(narrative_raw)
            structured = ParsedResumeDataStructured.model_validate(structured_raw)
            parsed = ParsedResumeData(
                contact=narrative.contact,
                summary=narrative.summary,
                skills=_drop_blank(narrative.skills),
                achievements=_drop_blank(narrative.achievements),
                languages=[lang for lang in narrative.languages if lang.language.strip()],
                experience=structured.experience,
                education=structured.education,
                projects=structured.projects,
                certifications=structured.certifications,
            )

            session.add(
                ResumeParsedData(
                    resume_version_id=version.id,
                    contact=parsed.contact.model_dump(),
                    summary=parsed.summary,
                    skills=parsed.skills,
                    experience=[item.model_dump() for item in parsed.experience],
                    education=[item.model_dump() for item in parsed.education],
                    projects=[item.model_dump() for item in parsed.projects],
                    certifications=[item.model_dump() for item in parsed.certifications],
                    achievements=parsed.achievements,
                    languages=[item.model_dump() for item in parsed.languages],
                    raw_text=text,
                    model_used=ai_service.model_name,
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
            await _advance_pipeline(
                user_id=user_id,
                resume_version_id=version_id,
                step="parsing",
                status="failed",
                error=str(exc)[:2000],
            )
            return

    await _advance_pipeline(
        user_id=user_id, resume_version_id=version_id, step="parsing", status="completed"
    )
    generate_resume_embedding_task.delay(str(version_id), str(user_id))


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


async def _generate_embedding(version_id: uuid.UUID, user_id: uuid.UUID) -> None:
    await _advance_pipeline(
        user_id=user_id, resume_version_id=version_id, step="embedding", status="running"
    )

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
            await _advance_pipeline(
                user_id=user_id,
                resume_version_id=version_id,
                step="embedding",
                status="failed",
                error="No parsed data found.",
            )
            return

        embedding_service = get_container().resolve("embedding_service")
        vector = await embedding_service.generate_embedding(text=_embedding_input(parsed))

        session.add(
            ResumeEmbedding(
                resume_version_id=version_id,
                embedding=vector,
                model_used=embedding_service.model_name,
            )
        )
        await session.commit()

    logger.info(
        "Resume embedding generated",
        extra={"extra_fields": {"version_id": str(version_id)}},
    )
    await _advance_pipeline(
        user_id=user_id, resume_version_id=version_id, step="embedding", status="completed"
    )

    calculate_resume_score_task.delay(str(version_id), str(user_id))


async def _calculate_resume_score(version_id: uuid.UUID, user_id: uuid.UUID) -> None:
    """Completeness/quality heuristic — no Claude call needed. Reuses
    `ats_resume_generator.scoring.structure_score` (has_summary/has_skills/
    has_experience/has_education/has_projects + quantified-bullet ratio).
    """
    from app.modules.ats_resume_generator import scoring as ats_scoring

    await _advance_pipeline(
        user_id=user_id, resume_version_id=version_id, step="scoring", status="running"
    )

    async with AsyncSessionLocal() as session:
        parsed = (
            await session.execute(
                select(ResumeParsedData).where(ResumeParsedData.resume_version_id == version_id)
            )
        ).scalars().one_or_none()
        if parsed is None:
            logger.warning(
                "No parsed data found for resume scoring",
                extra={"extra_fields": {"version_id": str(version_id)}},
            )
            await _advance_pipeline(
                user_id=user_id,
                resume_version_id=version_id,
                step="scoring",
                status="failed",
                error="No parsed data found.",
            )
            return

        bullets: list[str] = []
        for job in parsed.experience:
            bullets.extend(job.get("highlights") or [])

        result = ats_scoring.structure_score(
            has_summary=bool(parsed.summary),
            has_skills=bool(parsed.skills),
            has_experience=bool(parsed.experience),
            has_education=bool(parsed.education),
            has_projects=bool(parsed.projects),
            bullet_highlights=bullets,
        )

        repo = ResumeVersionRepository(session)
        version = await repo.get(version_id)
        if version is not None:
            await repo.update(
                version,
                resume_score=round(result.score, 2),
                resume_score_breakdown=result.checks,
            )
            await session.commit()

    logger.info(
        "Resume score calculated",
        extra={"extra_fields": {"version_id": str(version_id), "score": round(result.score, 2)}},
    )
    await _advance_pipeline(
        user_id=user_id, resume_version_id=version_id, step="scoring", status="completed"
    )

    run_ats_baseline_analysis_task.delay(str(version_id), str(user_id))


async def _top_skills_for_roles(session, *, target_roles: list[str], limit: int) -> list[str]:
    """Most common skills among active job postings whose title matches one
    of `target_roles` (substring match) — falls back to the overall most
    common skills among active postings when there are no target roles yet
    or the role filter returns nothing.
    """
    from app.modules.job_ingestion.models import Job, JobStatus, Skill, job_skill

    base_stmt = (
        select(Skill.name, func.count(func.distinct(Job.id)).label("job_count"))
        .select_from(Job)
        .join(job_skill, job_skill.c.job_id == Job.id)
        .join(Skill, Skill.id == job_skill.c.skill_id)
        .where(Job.status == JobStatus.ACTIVE, Job.deleted_at.is_(None))
    )

    role_filters = [Job.title.ilike(f"%{role.strip()}%") for role in target_roles if role.strip()]
    if role_filters:
        stmt = (
            base_stmt.where(or_(*role_filters))
            .group_by(Skill.name)
            .order_by(func.count(func.distinct(Job.id)).desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        rows = result.all()
        if rows:
            return [name for name, _ in rows]

    stmt = base_stmt.group_by(Skill.name).order_by(func.count(func.distinct(Job.id)).desc()).limit(limit)
    result = await session.execute(stmt)
    return [name for name, _ in result.all()]


async def _run_ats_baseline_analysis(version_id: uuid.UUID, user_id: uuid.UUID) -> None:
    """No-job-description ATS health check: structure score (reused from
    `_calculate_resume_score`'s companion heuristic) plus a skill-gap
    estimate against the most in-demand skills for the user's target roles.
    No Claude call — deterministic and idempotent.
    """
    from app.modules.ats_resume_generator import scoring as ats_scoring
    from app.modules.user_profile.repository import UserProfileRepository

    await _advance_pipeline(
        user_id=user_id, resume_version_id=version_id, step="ats_baseline", status="running"
    )

    async with AsyncSessionLocal() as session:
        parsed = (
            await session.execute(
                select(ResumeParsedData).where(ResumeParsedData.resume_version_id == version_id)
            )
        ).scalars().one_or_none()
        if parsed is None:
            logger.warning(
                "No parsed data found for ATS baseline analysis",
                extra={"extra_fields": {"version_id": str(version_id)}},
            )
            await _advance_pipeline(
                user_id=user_id,
                resume_version_id=version_id,
                step="ats_baseline",
                status="failed",
                error="No parsed data found.",
            )
            return

        profile = await UserProfileRepository(session).get_by_user(user_id=user_id)
        target_roles = profile.target_roles if profile else []
        top_skills = await _top_skills_for_roles(
            session, target_roles=target_roles, limit=_BASELINE_SKILL_LIMIT
        )

        resume_text = ats_scoring.build_resume_search_text(
            summary=parsed.summary,
            skills=parsed.skills,
            projects=parsed.projects,
            experience=parsed.experience,
            education=parsed.education,
            certifications=parsed.certifications,
        )
        keyword_result = ats_scoring.keyword_coverage(top_skills, resume_text)

        bullets: list[str] = []
        for job in parsed.experience:
            bullets.extend(job.get("highlights") or [])
        structure_result = ats_scoring.structure_score(
            has_summary=bool(parsed.summary),
            has_skills=bool(parsed.skills),
            has_experience=bool(parsed.experience),
            has_education=bool(parsed.education),
            has_projects=bool(parsed.projects),
            bullet_highlights=bullets,
        )

        settings = get_settings()
        ats_baseline_score = ats_scoring.compute_ats_score(
            keyword_score=keyword_result.score,
            structure_score_value=structure_result.score,
            keyword_weight=settings.ATS_KEYWORD_SCORE_WEIGHT,
            structure_weight=settings.ATS_STRUCTURE_SCORE_WEIGHT,
        )

        repo = ResumeVersionRepository(session)
        version = await repo.get(version_id)
        if version is not None:
            from datetime import datetime, timezone

            await repo.update(
                version,
                ats_baseline_score=round(ats_baseline_score, 2),
                ats_baseline_missing_skills=keyword_result.missing,
                health_computed_at=datetime.now(timezone.utc),
            )
            await session.commit()

    logger.info(
        "ATS baseline analysis completed",
        extra={
            "extra_fields": {
                "version_id": str(version_id),
                "ats_baseline_score": round(ats_baseline_score, 2),
            }
        },
    )
    await _advance_pipeline(
        user_id=user_id, resume_version_id=version_id, step="ats_baseline", status="completed"
    )

    from app.modules.matching.tasks import compute_resume_matches_task

    compute_resume_matches_task.delay(str(version_id))
