"""Matching module — application/use-case layer.

Orchestrates pgvector semantic similarity (via `MatchingRepository`), rule
boosts and narrative generation (via `app.modules.matching.scoring`, kept
pure/DB-free), persistence, and realtime notification. No embeddings are
generated here — resumes and jobs are embedded once, upstream, by
`resume_parser`/`job_ingestion`; this module only reads those vectors.
"""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from collections.abc import Sequence

from app.core.config import get_settings
from app.core.exceptions import ConflictError, NotFoundError
from app.core.logging import get_logger
from app.modules.job_ingestion.models import Job
from app.modules.matching import realtime, scoring
from app.modules.matching.models import (
    MatchPreference,
    MatchStatus,
    Recommendation,
    RecommendationType,
    ResumeJobMatch,
)
from app.modules.matching.repository import MatchingRepository
from app.modules.matching.schemas import (
    JobSummary,
    MatchPreferenceRequest,
    RefreshRecommendationsResponse,
    ResumeJobMatchRead,
    ScoreBreakdown,
    SkillCheckItem,
)
from app.modules.resume_parser.models import ResumeVersion

logger = get_logger(__name__)

# A match needs at least this score to be considered "recommendation-worthy"
# when aggregating company/skill recommendations from raw matches.
_RECOMMENDATION_FLOOR_SCORE = 50.0


class MatchingService:
    def __init__(self, repository: MatchingRepository) -> None:
        self.repository = repository
        self._settings = get_settings()

    # --- preferences --------------------------------------------------------------

    async def get_preferences(self, *, user_id: uuid.UUID) -> MatchPreference:
        preference = await self.repository.get_preference(user_id=user_id)
        if preference is None:
            raise NotFoundError("No match preferences have been set yet.")
        return preference

    async def upsert_preferences(
        self, *, user_id: uuid.UUID, data: MatchPreferenceRequest
    ) -> MatchPreference:
        return await self.repository.upsert_preference(
            user_id=user_id, **data.model_dump(mode="json")
        )

    # --- scoring core --------------------------------------------------------------

    def _score_pair(
        self,
        *,
        resume_version: ResumeVersion,
        job: Job,
        preference: MatchPreference | None,
        semantic_score: float,
    ) -> dict:
        settings = self._settings
        resume_skills = resume_version.parsed_data.skills if resume_version.parsed_data else []
        job_skill_pairs = [(skill.name, skill.normalized_name) for skill in job.skills]
        skill_result = scoring.skill_match(resume_skills, job_skill_pairs)

        if preference is not None and preference.is_active:
            boosts: dict[str, float | None] = {
                "country": scoring.country_boost(
                    preference.preferred_countries,
                    job.location.country if job.location else None,
                    cap=settings.MATCHING_BOOST_COUNTRY,
                ),
                "work_mode": scoring.work_mode_boost(
                    preference.preferred_work_modes,
                    job.work_mode,
                    cap=settings.MATCHING_BOOST_WORK_MODE,
                ),
                "employment_type": scoring.employment_type_boost(
                    preference.preferred_employment_types,
                    job.employment_type,
                    cap=settings.MATCHING_BOOST_EMPLOYMENT_TYPE,
                ),
                "salary": scoring.salary_boost(
                    expected_min=preference.expected_salary_min,
                    expected_max=preference.expected_salary_max,
                    expected_currency=preference.salary_currency,
                    job_min=job.salary_min,
                    job_max=job.salary_max,
                    job_currency=job.salary_currency,
                    cap=settings.MATCHING_BOOST_SALARY,
                ),
                "experience": scoring.experience_boost(
                    preference.years_of_experience,
                    job.title,
                    cap=settings.MATCHING_BOOST_EXPERIENCE,
                ),
            }
        else:
            boosts = {
                "country": None,
                "work_mode": None,
                "employment_type": None,
                "salary": None,
                "experience": None,
            }

        match_score = scoring.compute_match_score(
            semantic_score=semantic_score,
            skill_score=skill_result.score,
            boosts=boosts.values(),
            semantic_weight=settings.MATCHING_SEMANTIC_WEIGHT,
            skill_weight=settings.MATCHING_SKILL_WEIGHT,
        )
        narrative = scoring.build_strengths_and_weaknesses(
            semantic_score=semantic_score, skill_result=skill_result, boosts=boosts
        )
        rule_boost_score = sum(b for b in boosts.values() if b is not None)

        return {
            "status": MatchStatus.COMPUTED,
            "match_score": round(match_score, 2),
            "semantic_score": round(semantic_score, 5),
            "skill_match_score": round(skill_result.score, 2),
            "rule_boost_score": round(rule_boost_score, 2),
            "matched_skills": skill_result.matched,
            "missing_skills": skill_result.missing,
            "strengths": narrative.strengths,
            "weaknesses": narrative.weaknesses,
            "score_breakdown": {
                "semantic_score": semantic_score,
                "skill_match_score": skill_result.score,
                "rule_boost_score": rule_boost_score,
                "boosts": boosts,
            },
            "error_message": None,
        }

    def _to_read(self, match: ResumeJobMatch, job: Job) -> ResumeJobMatchRead:
        skill_checklist = [
            SkillCheckItem(skill=skill, matched=matched)
            for skill, matched in scoring.build_skill_checklist(
                job_skills=[s.name for s in job.skills],
                matched_skills=match.matched_skills,
                missing_skills=match.missing_skills,
            )
        ]
        return ResumeJobMatchRead(
            id=match.id,
            resume_version_id=match.resume_version_id,
            job_id=match.job_id,
            status=match.status,
            match_score=float(match.match_score) if match.match_score is not None else None,
            matched_skills=match.matched_skills,
            missing_skills=match.missing_skills,
            skill_checklist=skill_checklist,
            strengths=match.strengths,
            weaknesses=match.weaknesses,
            score_breakdown=ScoreBreakdown.model_validate(match.score_breakdown),
            ai_narrative=match.ai_narrative,
            job=JobSummary(
                id=job.id,
                title=job.title,
                company_name=job.company.name,
                location=job.location.raw_text if job.location else None,
                work_mode=job.work_mode,
                employment_type=job.employment_type,
                salary_min=float(job.salary_min) if job.salary_min is not None else None,
                salary_max=float(job.salary_max) if job.salary_max is not None else None,
                salary_currency=job.salary_currency,
                source_url=job.source_url,
                posted_at=job.posted_at,
            ),
            created_at=match.created_at,
            updated_at=match.updated_at,
        )

    # --- match computation -----------------------------------------------------------

    async def compute_match(
        self, *, user_id: uuid.UUID, resume_version_id: uuid.UUID, job_id: uuid.UUID
    ) -> ResumeJobMatchRead:
        """Compute (or recompute) a single resume<->job match inline.

        Bounded to one row, so this runs synchronously in the request rather
        than being queued — unlike `batch_compute_matches`, which fans out
        over up to `MATCHING_CANDIDATE_LIMIT` jobs and belongs on a worker.
        """
        resume_version = await self.repository.get_resume_version_owned(
            resume_version_id=resume_version_id, user_id=user_id
        )
        if resume_version is None:
            raise NotFoundError("Resume version not found.")
        if resume_version.embedding is None:
            raise ConflictError("This resume version's embedding isn't ready yet.")

        job = await self.repository.get_job(job_id=job_id)
        if job is None:
            raise NotFoundError("Job not found.")

        semantic_score = await self.repository.job_similarity(
            embedding=resume_version.embedding.embedding, job_id=job_id
        )
        if semantic_score is None:
            raise ConflictError("This job's embedding isn't ready yet.")

        preference = await self.repository.get_preference(user_id=user_id)
        fields = self._score_pair(
            resume_version=resume_version,
            job=job,
            preference=preference,
            semantic_score=semantic_score,
        )
        match = await self.repository.upsert_match(
            resume_version_id=resume_version_id, job_id=job_id, user_id=user_id, **fields
        )
        await realtime.publish_match_event(
            user_id,
            {
                "type": "match_computed",
                "match_id": str(match.id),
                "job_id": str(job_id),
                "match_score": fields["match_score"],
            },
        )
        return self._to_read(match, job)

    async def enqueue_batch_compute(
        self, *, user_id: uuid.UUID, resume_version_id: uuid.UUID
    ) -> None:
        """Validate ownership/readiness, then hand off to the `matching`
        Celery queue. The task import is local to avoid a module-import
        cycle with `tasks.py` (which imports this service) — same pattern
        as `job_ingestion.service.CompanyService._enqueue_fetch`.
        """
        resume_version = await self.repository.get_resume_version_owned(
            resume_version_id=resume_version_id, user_id=user_id
        )
        if resume_version is None:
            raise NotFoundError("Resume version not found.")
        if resume_version.embedding is None:
            raise ConflictError("This resume version's embedding isn't ready yet.")

        from app.modules.matching.tasks import compute_resume_matches_task

        compute_resume_matches_task.delay(str(resume_version_id))

    async def enqueue_recommendations_refresh(self, *, user_id: uuid.UUID) -> None:
        from app.modules.matching.tasks import refresh_recommendations_task

        refresh_recommendations_task.delay(str(user_id))

    async def batch_compute_matches(
        self, *, user_id: uuid.UUID, resume_version_id: uuid.UUID
    ) -> int:
        """Match a resume against its top `MATCHING_CANDIDATE_LIMIT` semantic
        candidates. Used by the `matching.compute_resume_matches` Celery task;
        does not publish per-row (the task publishes one aggregate event).
        """
        resume_version = await self.repository.get_resume_version_owned(
            resume_version_id=resume_version_id, user_id=user_id
        )
        if resume_version is None:
            raise NotFoundError("Resume version not found.")
        if resume_version.embedding is None:
            raise ConflictError("This resume version's embedding isn't ready yet.")

        preference = await self.repository.get_preference(user_id=user_id)
        candidates = await self.repository.semantic_candidate_jobs(
            embedding=resume_version.embedding.embedding,
            limit=self._settings.MATCHING_CANDIDATE_LIMIT,
        )

        count = 0
        for job, semantic_score in candidates:
            fields = self._score_pair(
                resume_version=resume_version,
                job=job,
                preference=preference,
                semantic_score=semantic_score,
            )
            await self.repository.upsert_match(
                resume_version_id=resume_version_id, job_id=job.id, user_id=user_id, **fields
            )
            count += 1
        return count

    async def list_matches(
        self, *, user_id: uuid.UUID, resume_version_id: uuid.UUID, offset: int, limit: int
    ) -> tuple[list[ResumeJobMatchRead], int]:
        matches, total = await self.repository.list_matches_for_resume(
            resume_version_id=resume_version_id, user_id=user_id, offset=offset, limit=limit
        )
        jobs_by_id = await self.repository.get_jobs_by_ids([m.job_id for m in matches])
        reads = [self._to_read(m, jobs_by_id[m.job_id]) for m in matches if m.job_id in jobs_by_id]
        return reads, total

    async def get_match(self, *, user_id: uuid.UUID, match_id: uuid.UUID) -> ResumeJobMatchRead:
        match = await self.repository.get_match(match_id=match_id, user_id=user_id)
        if match is None:
            raise NotFoundError("Match not found.")
        job = await self.repository.get_job(job_id=match.job_id)
        if job is None:
            raise NotFoundError("The job for this match no longer exists.")
        return self._to_read(match, job)

    # --- recommendations ---------------------------------------------------------------

    async def refresh_recommendations(
        self, *, user_id: uuid.UUID
    ) -> RefreshRecommendationsResponse:
        matches = await self.repository.list_matches_for_user(
            user_id=user_id, limit=self._settings.MATCHING_CANDIDATE_LIMIT
        )
        strong_matches = [m for m in matches if (m.match_score or 0) >= _RECOMMENDATION_FLOOR_SCORE]

        jobs_count = await self._refresh_job_recommendations(user_id, matches)
        companies_count = await self._refresh_company_recommendations(user_id, strong_matches)
        skills_count = await self._refresh_skill_recommendations(user_id, strong_matches)

        response = RefreshRecommendationsResponse(
            jobs=jobs_count, companies=companies_count, skills=skills_count
        )
        await realtime.publish_match_event(
            user_id, {"type": "recommendations_updated", **response.model_dump()}
        )
        return response

    async def _refresh_job_recommendations(
        self, user_id: uuid.UUID, matches: Sequence[ResumeJobMatch]
    ) -> int:
        ranked = sorted(matches, key=lambda m: m.match_score or 0, reverse=True)
        top = ranked[: self._settings.MATCHING_RECOMMENDATION_JOB_LIMIT]
        candidates = [
            {
                "job_id": m.job_id,
                "score": m.match_score or 0,
                "reason": m.strengths[0] if m.strengths else "Strong overall match score.",
            }
            for m in top
        ]
        return await self.repository.replace_active_recommendations(
            user_id=user_id, recommendation_type=RecommendationType.JOB, candidates=candidates
        )

    async def _refresh_company_recommendations(
        self, user_id: uuid.UUID, strong_matches: Sequence[ResumeJobMatch]
    ) -> int:
        jobs_by_id = await self.repository.get_jobs_by_ids([m.job_id for m in strong_matches])
        scores_by_company: dict[uuid.UUID, list[float]] = defaultdict(list)
        names_by_company: dict[uuid.UUID, str] = {}
        for match in strong_matches:
            job = jobs_by_id.get(match.job_id)
            if job is None:
                continue
            scores_by_company[job.company_id].append(match.match_score or 0)
            names_by_company[job.company_id] = job.company.name

        ranked = sorted(
            scores_by_company.items(), key=lambda kv: sum(kv[1]) / len(kv[1]), reverse=True
        )
        top = ranked[: self._settings.MATCHING_RECOMMENDATION_COMPANY_LIMIT]
        candidates = [
            {
                "company_id": company_id,
                "score": round(sum(scores) / len(scores), 2),
                "reason": f"{len(scores)} strong match(es) at {names_by_company[company_id]}.",
            }
            for company_id, scores in top
        ]
        return await self.repository.replace_active_recommendations(
            user_id=user_id, recommendation_type=RecommendationType.COMPANY, candidates=candidates
        )

    async def _refresh_skill_recommendations(
        self, user_id: uuid.UUID, strong_matches: Sequence[ResumeJobMatch]
    ) -> int:
        counts: Counter[str] = Counter()
        for match in strong_matches:
            for skill_name in match.missing_skills:
                counts[skill_name.strip().lower()] += 1

        skill_limit = self._settings.MATCHING_RECOMMENDATION_SKILL_LIMIT
        top_keys = [key for key, _ in counts.most_common(skill_limit)]
        skills_by_name = await self.repository.get_skills_by_normalized_names(top_keys)
        candidates = [
            {
                "skill_id": skills_by_name[key].id,
                "score": float(counts[key]),
                "reason": (
                    f"Missing from your resume but appears in {counts[key]} of your "
                    "matched job posting(s)."
                ),
            }
            for key in top_keys
            if key in skills_by_name
        ]
        return await self.repository.replace_active_recommendations(
            user_id=user_id, recommendation_type=RecommendationType.SKILL, candidates=candidates
        )

    async def list_recommendations(
        self, *, user_id: uuid.UUID, recommendation_type: RecommendationType
    ) -> list[dict]:
        recommendations = await self.repository.list_recommendations(
            user_id=user_id, recommendation_type=recommendation_type
        )
        display_names = await self._resolve_display_names(recommendation_type, recommendations)
        return [
            {
                "id": r.id,
                "recommendation_type": r.recommendation_type,
                "job_id": r.job_id,
                "company_id": r.company_id,
                "skill_id": r.skill_id,
                "display_name": (
                    display_names.get(target_id, "Unknown")
                    if (target_id := _target_id(r)) is not None
                    else "Unknown"
                ),
                "score": float(r.score),
                "reason": r.reason,
                "status": r.status,
                "created_at": r.created_at,
            }
            for r in recommendations
        ]

    async def _resolve_display_names(
        self, recommendation_type: RecommendationType, recommendations: Sequence[Recommendation]
    ) -> dict[uuid.UUID, str]:
        ids = [target_id for r in recommendations if (target_id := _target_id(r)) is not None]
        if recommendation_type == RecommendationType.JOB:
            jobs = await self.repository.get_jobs_by_ids(ids)
            return {job_id: job.title for job_id, job in jobs.items()}
        if recommendation_type == RecommendationType.COMPANY:
            companies = await self.repository.get_companies_by_ids(ids)
            return {company_id: company.name for company_id, company in companies.items()}
        skills = await self.repository.get_skills_by_ids(ids)
        return {skill_id: skill.name for skill_id, skill in skills.items()}

    async def dismiss_recommendation(
        self, *, user_id: uuid.UUID, recommendation_id: uuid.UUID
    ) -> Recommendation:
        recommendation = await self.repository.get_recommendation(
            recommendation_id=recommendation_id, user_id=user_id
        )
        if recommendation is None:
            raise NotFoundError("Recommendation not found.")
        return await self.repository.dismiss_recommendation(recommendation)


def _target_id(recommendation: Recommendation) -> uuid.UUID | None:
    return recommendation.job_id or recommendation.company_id or recommendation.skill_id
