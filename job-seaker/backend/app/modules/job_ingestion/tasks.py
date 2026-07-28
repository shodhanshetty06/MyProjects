"""Job Ingestion module — Celery tasks.

Five tasks forming the ingestion pipeline, routed to three queues (see
`app/workers/celery_app.py` `task_routes` and the `worker-ingestion` /
`worker-embedding` services in `docker-compose.yml`):

    sweep_companies_task     -> "ingestion" queue (Celery Beat, periodic)
    fetch_company_jobs_task  -> "ingestion" queue
    normalize_job_task       -> "ingestion" queue
    extract_job_skills_task  -> "skills" queue
    generate_job_embedding_task -> "embedding" queue (shared with resume_parser)

Each is a thin sync wrapper around `asyncio.run(...)`, using `AsyncSessionLocal`
directly — same shape as `resume_parser/tasks.py`. `fetch_company_jobs_task`
is the only stage that talks to the outside world (the ATS adapter), so it's
the only one with `autoretry_for` configured for retry-on-transient-failure.
"""

from __future__ import annotations

import uuid
from asyncio import run as run_async
from datetime import UTC, datetime, timedelta

import httpx
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.exceptions import ServiceUnavailableError
from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal
from app.dependency_injection.container import get_container
from app.integrations.http_client import build_http_client
from app.modules.job_ingestion.adapters import get_adapter
from app.modules.job_ingestion.models import (
    IngestionRunStatus,
    Job,
    JobEmbedding,
    JobStatus,
    SeniorityLevel,
    WorkMode,
)
from app.modules.job_ingestion.normalization import (
    clean_title,
    compute_content_hash,
    parse_employment_type,
    parse_location,
    parse_salary,
    parse_work_mode,
    validate_posting,
)
from app.modules.job_ingestion.quality import compute_quality_score
from app.modules.job_ingestion.repository import (
    CompanyIntelligenceRepository,
    CompanyRepository,
    IngestionRunRepository,
    JobRepository,
    LocationRepository,
    SkillRepository,
)
from app.workers.base_task import BaseTask
from app.workers.celery_app import celery_app

logger = get_logger(__name__)

# "AI Job Understanding" (Feature 3) — a single structured-extraction call
# pulls out everything downstream matching/quality-scoring/display need, so
# one round trip to the LLM covers skills, responsibilities, benefits, tech
# stack, seniority, industry, and department instead of five separate ones.
JOB_UNDERSTANDING_SYSTEM_PROMPT = (
    "You are a job posting analyst. Read the job posting text and extract only what is "
    "explicitly stated or unambiguously implied — never invent details the text doesn't "
    "support. If a field has no support in the text, return an empty list/null for it "
    "rather than guessing.\n\n"
    "- required_skills: concrete technical/professional skills the posting describes as "
    "mandatory. Short canonical names (e.g. 'Python', 'Project Management').\n"
    "- preferred_skills: skills described as a plus/nice-to-have, not required.\n"
    "- responsibilities: short phrases describing what the person will actually do day to "
    "day (not qualifications).\n"
    "- benefits: concrete perks/benefits mentioned (e.g. 'Health insurance', '401k match', "
    "'Unlimited PTO').\n"
    "- tech_stack: specific named tools/languages/frameworks/platforms mentioned.\n"
    "- seniority: one of intern, junior, mid, senior, lead, staff, principal, executive, or "
    "unknown if the text gives no seniority signal.\n"
    "- industry: the company's industry/sector if stated or clearly implied, else null.\n"
    "- department: the team/department this role sits in (e.g. 'Engineering', 'Sales'), "
    "else null."
)

_VALID_SENIORITY = {level.value for level in SeniorityLevel}


class _JobUnderstanding(BaseModel):
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    benefits: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    seniority: str = SeniorityLevel.UNKNOWN.value
    industry: str | None = None
    department: str | None = None


@celery_app.task(base=BaseTask, bind=True, name="job_ingestion.sweep_companies")
def sweep_companies_task(self) -> None:  # noqa: ANN001 -- Celery `self`
    run_async(_sweep_companies())


@celery_app.task(
    base=BaseTask,
    bind=True,
    name="job_ingestion.fetch_company_jobs",
    autoretry_for=(httpx.HTTPError, ServiceUnavailableError),
)
def fetch_company_jobs_task(self, company_id: str) -> None:  # noqa: ANN001
    run_async(_fetch_company_jobs(uuid.UUID(company_id), retry_count=self.request.retries))


@celery_app.task(base=BaseTask, bind=True, name="job_ingestion.normalize_job")
def normalize_job_task(self, job_id: str) -> None:  # noqa: ANN001
    run_async(_normalize_job(uuid.UUID(job_id)))


@celery_app.task(base=BaseTask, bind=True, name="job_ingestion.extract_job_skills")
def extract_job_skills_task(self, job_id: str) -> None:  # noqa: ANN001
    run_async(_extract_job_skills(uuid.UUID(job_id)))


@celery_app.task(base=BaseTask, bind=True, name="job_ingestion.generate_job_embedding")
def generate_job_embedding_task(self, job_id: str) -> None:  # noqa: ANN001
    run_async(_generate_job_embedding(uuid.UUID(job_id)))


async def _sweep_companies() -> None:
    settings = get_settings()
    stale_before = datetime.now(UTC) - timedelta(hours=settings.JOB_INGESTION_STALE_CRAWL_HOURS)

    async with AsyncSessionLocal() as session:
        company_repo = CompanyRepository(session)
        companies = await company_repo.list_due_for_crawl(stale_before=stale_before)
        company_ids = [company.id for company in companies]

    for company_id in company_ids:
        fetch_company_jobs_task.delay(str(company_id))

    logger.info(
        "Job ingestion sweep enqueued",
        extra={"extra_fields": {"company_count": len(company_ids)}},
    )


async def _fetch_company_jobs(company_id: uuid.UUID, *, retry_count: int = 0) -> None:
    settings = get_settings()

    async with AsyncSessionLocal() as session:
        company_repo = CompanyRepository(session)
        intelligence_repo = CompanyIntelligenceRepository(session)
        job_repo = JobRepository(session)
        run_repo = IngestionRunRepository(session)

        company = await company_repo.get_with_intelligence(company_id)
        if company is None or company.intelligence is None:
            logger.warning(
                "Company or intelligence config not found for ingestion",
                extra={"extra_fields": {"company_id": str(company_id)}},
            )
            return

        intelligence = company.intelligence
        run = await run_repo.create(
            company_id=company.id,
            source=intelligence.ats_platform,
            status=IngestionRunStatus.RUNNING,
            started_at=datetime.now(UTC),
            retry_count=retry_count,
        )
        await session.commit()

        try:
            async with build_http_client() as http_client:
                adapter = get_adapter(
                    intelligence.ats_platform,
                    ats_identifier=intelligence.ats_identifier,
                    career_url=intelligence.career_url,
                    adapter_config=intelligence.adapter_config,
                    http_client=http_client,
                )
                postings = await adapter.fetch_jobs()

            postings = postings[: settings.JOB_INGESTION_MAX_JOBS_PER_RUN]
            now = datetime.now(UTC)

            created = updated = failed = 0
            changed_job_ids: list[uuid.UUID] = []

            for posting in postings:
                try:
                    title = clean_title(posting.title)
                    # Data Cleaning Layer (Feature 2): validate before this
                    # posting ever reaches the database. A bad title, an
                    # empty description, or a missing URL means the upstream
                    # ATS/adapter served something unusable — skip it with a
                    # clear reason instead of storing garbage.
                    validation = validate_posting(
                        title=title, description=posting.description_text, url=posting.url
                    )
                    if not validation.is_valid:
                        failed += 1
                        logger.warning(
                            "Skipped invalid job posting",
                            extra={
                                "extra_fields": {
                                    "company_id": str(company_id),
                                    "external_id": posting.external_id,
                                    "errors": validation.errors,
                                }
                            },
                        )
                        continue

                    content_hash = compute_content_hash(
                        title=title,
                        description=posting.description_text,
                        location_raw=posting.location_raw,
                        salary_raw=posting.salary_raw,
                    )
                    raw_payload = {
                        "adapter_raw": posting.raw,
                        "location_raw": posting.location_raw,
                        "employment_type_raw": posting.employment_type_raw,
                        "remote": posting.remote,
                        "description_html": posting.description_html,
                        "raw_title": posting.title,
                    }

                    existing = await job_repo.get_by_source(
                        source=intelligence.ats_platform, source_job_id=posting.external_id
                    )
                    if existing is None:
                        job = await job_repo.create(
                            company_id=company.id,
                            source=intelligence.ats_platform,
                            source_job_id=posting.external_id,
                            source_url=posting.url,
                            title=title,
                            description=posting.description_text,
                            status=JobStatus.PENDING,
                            content_hash=content_hash,
                            raw_payload=raw_payload,
                            salary_raw_text=posting.salary_raw,
                            posted_at=posting.posted_at,
                            last_seen_at=now,
                        )
                        created += 1
                        changed_job_ids.append(job.id)
                    elif existing.content_hash != content_hash:
                        await job_repo.update(
                            existing,
                            title=title,
                            source_url=posting.url,
                            description=posting.description_text,
                            content_hash=content_hash,
                            raw_payload=raw_payload,
                            salary_raw_text=posting.salary_raw,
                            posted_at=posting.posted_at,
                            status=JobStatus.PENDING,
                            last_seen_at=now,
                        )
                        updated += 1
                        changed_job_ids.append(existing.id)
                    else:
                        await job_repo.update(existing, last_seen_at=now)
                except Exception as exc:  # noqa: BLE001 -- one bad posting must not abort the run
                    failed += 1
                    logger.error(
                        "Failed to ingest a job posting",
                        extra={
                            "extra_fields": {
                                "company_id": str(company_id),
                                "external_id": posting.external_id,
                                "error": str(exc),
                            }
                        },
                    )

            closed = await job_repo.bulk_close_stale(company_id=company.id, seen_before=now)

            run_status = IngestionRunStatus.SUCCESS if failed == 0 else IngestionRunStatus.PARTIAL
            await run_repo.update(
                run,
                status=run_status,
                finished_at=datetime.now(UTC),
                jobs_found=len(postings),
                jobs_created=created,
                jobs_updated=updated,
                jobs_closed=closed,
                jobs_failed=failed,
            )
            await intelligence_repo.update(
                intelligence,
                last_crawled_at=now,
                last_crawl_status=run_status,
                hiring_frequency_per_month=float(created),
            )
            await session.commit()
        except Exception as exc:
            await session.rollback()
            await run_repo.update(
                run,
                status=IngestionRunStatus.FAILED,
                finished_at=datetime.now(UTC),
                error_message=str(exc)[:2000],
            )
            await intelligence_repo.update(
                intelligence,
                last_crawled_at=datetime.now(UTC),
                last_crawl_status=IngestionRunStatus.FAILED,
            )
            await session.commit()
            logger.error(
                "Company job ingestion failed",
                extra={"extra_fields": {"company_id": str(company_id), "error": str(exc)}},
            )
            raise

    for job_id in changed_job_ids:
        normalize_job_task.delay(str(job_id))


async def _normalize_job(job_id: uuid.UUID) -> None:
    async with AsyncSessionLocal() as session:
        job_repo = JobRepository(session)
        location_repo = LocationRepository(session)

        job = await job_repo.get(job_id)
        if job is None:
            logger.warning(
                "Job not found for normalization", extra={"extra_fields": {"job_id": str(job_id)}}
            )
            return

        await job_repo.update(job, status=JobStatus.NORMALIZING)
        await session.commit()

        try:
            meta = job.raw_payload or {}
            location_raw = meta.get("location_raw")
            employment_type_raw = meta.get("employment_type_raw")
            remote_flag = meta.get("remote")

            salary = parse_salary(job.salary_raw_text)
            employment_type = parse_employment_type(employment_type_raw)
            work_mode = parse_work_mode(
                location_raw=location_raw, remote_flag=remote_flag, description=job.description
            )
            location_parts = parse_location(location_raw)
            if work_mode == WorkMode.UNKNOWN and location_parts.is_remote:
                work_mode = WorkMode.REMOTE

            location = None
            if not location_parts.is_remote and location_parts.city:
                normalized_key = "|".join(
                    [
                        (location_parts.city or "").lower(),
                        (location_parts.region or "").lower(),
                        (location_parts.country or "").lower(),
                    ]
                )
                location = await location_repo.get_or_create(
                    normalized_key=normalized_key,
                    raw_text=location_raw or "",
                    city=location_parts.city,
                    region=location_parts.region,
                    country=location_parts.country,
                    country_code=location_parts.country_code,
                )

            await job_repo.update(
                job,
                employment_type=employment_type,
                work_mode=work_mode,
                salary_min=salary.salary_min,
                salary_max=salary.salary_max,
                salary_currency=salary.currency,
                salary_period=salary.period,
                location_id=location.id if location else None,
                status=JobStatus.ACTIVE,
                error_message=None,
            )
            await session.commit()
        except Exception as exc:
            await session.rollback()
            await job_repo.update(job, status=JobStatus.FAILED, error_message=str(exc)[:2000])
            await session.commit()
            logger.error(
                "Job normalization failed",
                extra={"extra_fields": {"job_id": str(job_id), "error": str(exc)}},
            )
            return

    from app.events.bus import event_bus
    from app.events.domain import JobNormalized

    await event_bus.publish(JobNormalized(job_id=job_id))


async def _extract_job_skills(job_id: uuid.UUID) -> None:
    async with AsyncSessionLocal() as session:
        job_repo = JobRepository(session)
        skill_repo = SkillRepository(session)
        company_repo = CompanyRepository(session)

        job = await job_repo.get_with_details(job_id)
        if job is None:
            logger.warning(
                "Job not found for skill extraction",
                extra={"extra_fields": {"job_id": str(job_id)}},
            )
            return

        try:
            ai_service = get_container().resolve("ai_service")
            extracted = await ai_service.extract_structured(
                text=f"{job.title}\n\n{job.description}",
                schema=_JobUnderstanding.model_json_schema(),
                system=JOB_UNDERSTANDING_SYSTEM_PROMPT,
            )
            understanding = _JobUnderstanding.model_validate(extracted)
            seniority = (
                understanding.seniority
                if understanding.seniority in _VALID_SENIORITY
                else SeniorityLevel.UNKNOWN.value
            )
            # `skills` (the search/matching relationship) stays the union of
            # required+preferred so existing skill-based search/matching
            # keeps working unchanged; the finer breakdown lives in the new
            # columns below for display.
            all_skill_names = list(
                dict.fromkeys(understanding.required_skills + understanding.preferred_skills)
            )
            job.skills = await skill_repo.bulk_get_or_create(names=all_skill_names)
            await job_repo.update(
                job,
                required_skills=understanding.required_skills,
                preferred_skills=understanding.preferred_skills,
                responsibilities=understanding.responsibilities,
                benefits=understanding.benefits,
                tech_stack=understanding.tech_stack,
                seniority=seniority,
                industry=understanding.industry,
                department=understanding.department,
            )
            await session.commit()
        except Exception as exc:
            await session.rollback()
            logger.error(
                "AI job understanding failed — continuing pipeline without it",
                extra={"extra_fields": {"job_id": str(job_id), "error": str(exc)}},
            )

        # Job Quality Score (Feature 5) — deterministic and independent of
        # whether the AI step above succeeded, so a scored job never depends
        # on LLM availability. Runs after the AI step (when it succeeds) so
        # the "skills identified" signal reflects the fuller extraction.
        try:
            job = await job_repo.get_with_details(job_id)
            company = await company_repo.get_with_intelligence(job.company_id)
            if job is not None and company is not None:
                result = compute_quality_score(job, company, settings=get_settings())
                await job_repo.update(
                    job, quality_score=result.score, quality_reasons=result.reasons_as_dicts()
                )
                await session.commit()
        except Exception as exc:
            await session.rollback()
            logger.error(
                "Job quality scoring failed",
                extra={"extra_fields": {"job_id": str(job_id), "error": str(exc)}},
            )

    generate_job_embedding_task.delay(str(job_id))


def _embedding_input(job: Job) -> str:
    parts = [job.title, job.company.name if job.company else "", job.description]
    if job.skills:
        parts.append("Skills: " + ", ".join(skill.name for skill in job.skills))
    if job.tech_stack:
        parts.append("Tech stack: " + ", ".join(job.tech_stack))
    return "\n".join(part for part in parts if part)


async def _generate_job_embedding(job_id: uuid.UUID) -> None:
    async with AsyncSessionLocal() as session:
        job_repo = JobRepository(session)
        job = await job_repo.get_with_details(job_id)
        if job is None:
            logger.warning(
                "Job not found for embedding generation",
                extra={"extra_fields": {"job_id": str(job_id)}},
            )
            return

        embedding_service = get_container().resolve("embedding_service")
        vector = await embedding_service.generate_embedding(text=_embedding_input(job))

        if job.embedding is not None:
            await session.delete(job.embedding)
            await session.flush()

        session.add(
            JobEmbedding(job_id=job.id, embedding=vector, model_used=embedding_service.model_name)
        )
        await session.commit()

    logger.info("Job embedding generated", extra={"extra_fields": {"job_id": str(job_id)}})

    # Smart Trigger: New Job Ingested -> match against affected users ->
    # notify users whose score clears the threshold. Both skills (extracted
    # earlier in this chain) and the embedding (just written above) are
    # ready by this point. Published as a domain event (see
    # `app/events/handlers/job_pipeline.py`) rather than calling
    # `match_new_job_for_active_users_task.delay()` directly, so the
    # matching module doesn't have to be imported here.
    from app.events.bus import event_bus
    from app.events.domain import JobEmbedded

    await event_bus.publish(JobEmbedded(job_id=job_id))
