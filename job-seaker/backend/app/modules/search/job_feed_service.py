"""Search module — Job Feed composition service (the "no duplicated logic"
core of the Job Search redesign).

Composes the personalized job feed, "why this score?" explanation, and home
summary entirely from other modules' already-existing repositories/services
— `matching` for scores/recommendations, `tracking` for application status,
`workspace` for the shared "next best action" list — the same
composition-over-reimplementation pattern `WorkspaceService` already uses.
The only *new* logic here is `heuristics.py`'s pure v1 signals and the
thin glue that assembles a `JobFeedItem` from all of the above.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.modules.matching.repository import MatchingRepository
from app.modules.matching.scoring import build_skill_checklist
from app.modules.search import heuristics, ranking
from app.modules.search.repository import SavedJobRepository, SearchRepository
from app.modules.search.schemas import (
    JobAISignals,
    JobFeedFilters,
    JobFeedItem,
    JobFeedMode,
    JobMatchContext,
    JobSearchHomeSummary,
    SavedJobRead,
    SearchFilters,
    SearchSortBy,
    SkillCheckItem,
    SortDirection,
    WhyThisScorePanel,
)
from app.modules.search.service import SearchService
from app.modules.tracking.repository import ApplicationRepository
from app.schemas.common import PaginationParams

logger = get_logger(__name__)

_RESUME_SUGGESTION_LIMIT = 5


def _build_greeting(now: datetime) -> str:
    """Pure function of the server's UTC hour — documented limitation (see
    plan section 12): no per-user timezone field exists in the schema yet.
    """
    hour = now.hour
    if 5 <= hour < 12:
        return "Good morning"
    if 12 <= hour < 18:
        return "Good afternoon"
    if 18 <= hour < 22:
        return "Good evening"
    return "Good night"


def _describe_boost(value: float | None) -> str | None:
    """Turn one `ResumeJobMatch.score_breakdown["boosts"]` entry into a
    human-readable fragment for `WhyThisScorePanel`. `None` means the user
    never set a preference for that dimension (boost wasn't computed at
    all, distinct from a computed-but-zero boost).
    """
    if value is None:
        return "No preference set for this."
    if value > 0:
        return f"Matches your preference (+{value:.1f} pts)."
    return "Does not match your preference."


class SavedJobService:
    """Save/unsave a job — idempotent get-or-create, not upsert-with-error,
    since "save an already-saved job" is a normal double-click, not a
    conflict the caller needs to handle.
    """

    def __init__(
        self, *, repository: SavedJobRepository, search_repository: SearchRepository
    ) -> None:
        self._repository = repository
        self._search_repository = search_repository

    async def save(self, *, user_id: uuid.UUID, job_id: uuid.UUID) -> SavedJobRead:
        from app.modules.search import realtime

        existing = await self._repository.get_by_user_and_job(user_id=user_id, job_id=job_id)
        if existing is not None:
            return SavedJobRead.model_validate(existing)

        jobs_by_id = await self._search_repository.fetch_jobs_by_ids([job_id])
        if job_id not in jobs_by_id:
            raise NotFoundError("Job not found.")

        saved = await self._repository.create(user_id=user_id, job_id=job_id)
        await realtime.publish_search_event(
            user_id, {"type": "job_saved", "job_id": str(job_id)}
        )
        return SavedJobRead.model_validate(saved)

    async def unsave(self, *, user_id: uuid.UUID, job_id: uuid.UUID) -> None:
        from app.modules.search import realtime

        deleted = await self._repository.delete_by_user_and_job(user_id=user_id, job_id=job_id)
        if not deleted:
            raise NotFoundError("Saved job not found.")
        await realtime.publish_search_event(
            user_id, {"type": "job_unsaved", "job_id": str(job_id)}
        )


class JobFeedService:
    """Composition root for the personalized Job Search home page."""

    def __init__(
        self,
        *,
        search_service: SearchService,
        search_repository: SearchRepository,
        saved_job_repository: SavedJobRepository,
        matching_repository: MatchingRepository,
        application_repository: ApplicationRepository,
        session: AsyncSession,
    ) -> None:
        self._search_service = search_service
        self._search_repository = search_repository
        self._saved_job_repository = saved_job_repository
        self._matching_repository = matching_repository
        self._application_repository = application_repository
        # Held only to build a `WorkspaceService` on demand in
        # `get_home_summary` (see its docstring) — every other method here
        # goes through the injected repositories/services above, not a raw
        # session, keeping this module's own queries testable/mockable.
        self._session = session
        self._settings = get_settings()

    # --- feed ------------------------------------------------------------------------

    async def get_feed(
        self, *, user_id: uuid.UUID, filters: JobFeedFilters, pagination: PaginationParams
    ) -> tuple[list[JobFeedItem], int]:
        mode = filters.feed_mode
        if mode == JobFeedMode.RECOMMENDED:
            return await self._feed_recommended(user_id, filters, pagination)
        if mode == JobFeedMode.SAVED:
            return await self._feed_saved(user_id, pagination)
        if mode == JobFeedMode.APPLIED:
            return await self._feed_applied(user_id, pagination)
        if mode == JobFeedMode.TRENDING:
            return await self._feed_trending(user_id, pagination)
        return await self._feed_via_search(user_id, mode, filters, pagination)

    async def _feed_recommended(
        self, user_id: uuid.UUID, filters: JobFeedFilters, pagination: PaginationParams
    ) -> tuple[list[JobFeedItem], int]:
        """Top candidate matches re-sorted by the Ranking Engine's weighted
        composite (`search/ranking.py`) — Resume Match 40% + Hiring
        Probability 20% + Salary 15% + Recency 10% + Company Quality 10% +
        Competition 5% — instead of plain `match_score DESC`, so "first job
        is actually the best opportunity," not just the closest skill match.
        If the user has zero computed matches yet (fresh signup, matching
        pipeline still running), falls back to `newest` so the page is
        never an empty state.
        """
        matches = await self._matching_repository.list_matches_for_user(
            user_id=user_id, limit=self._settings.MATCHING_CANDIDATE_LIMIT
        )
        if not matches:
            return await self._feed_via_search(user_id, JobFeedMode.NEWEST, filters, pagination)

        # Re-rank happens over the *whole* candidate set (bounded by
        # MATCHING_CANDIDATE_LIMIT, already the size `matches` was fetched
        # at) before slicing to a page — otherwise pagination would lock in
        # the old match_score-only order and only re-sort within one page.
        jobs_by_id = await self._search_repository.fetch_jobs_by_ids(
            [m.job_id for m in matches]
        )
        matches = [m for m in matches if m.job_id in jobs_by_id]
        rankings = await self._compute_rankings(matches, jobs_by_id)
        matches.sort(key=lambda m: rankings[m.job_id].score, reverse=True)

        total = len(matches)
        page_matches = matches[pagination.offset : pagination.offset + pagination.page_size]
        pairs = [
            (self._search_service.to_result(jobs_by_id[m.job_id], None), m) for m in page_matches
        ]
        items = await self._to_feed_items(user_id, pairs)
        for item in items:
            breakdown = rankings.get(item.id)
            if breakdown is not None:
                item.ranking_breakdown = {
                    "resume_match": breakdown.resume_match,
                    "hiring_probability": breakdown.hiring_probability,
                    "salary": breakdown.salary,
                    "recency": breakdown.recency,
                    "company_quality": breakdown.company_quality,
                    "competition": breakdown.competition,
                    "score": breakdown.score,
                }
        return items, total

    async def _compute_rankings(
        self, matches: list, jobs_by_id: dict
    ) -> dict[uuid.UUID, ranking.RankingBreakdown]:
        """Batch-compute the Ranking Engine's per-job composite score for a
        candidate set. One aggregate query per signal (hiring frequency,
        company quality, competition count) for the whole set, not per job —
        matching this module's existing batching convention
        (`get_hiring_frequency_by_company_ids` et al.).
        """
        settings = self._settings
        now = datetime.now(UTC)
        decay_days = settings.SEARCH_HIRING_PROB_RECENCY_DECAY_DAYS

        jobs = list(jobs_by_id.values())
        company_ids = [j.company_id for j in jobs]
        job_ids = [j.id for j in jobs]

        hiring_freq_by_company = await self._search_repository.get_hiring_frequency_by_company_ids(
            company_ids
        )
        company_rating_by_company = await self._search_repository.get_company_quality_by_company_ids(
            company_ids
        )
        avg_job_quality_by_company = await self._search_repository.get_avg_job_quality_by_company_ids(
            company_ids
        )
        competition_by_job = await self._search_repository.get_competition_counts(
            job_ids, min_match_score=settings.RANKING_COMPETITION_MATCH_THRESHOLD
        )
        comparison_salaries = [
            float(j.salary_max) for j in jobs if j.salary_max is not None
        ]

        weights = ranking.RankingWeights(
            resume_match=settings.RANKING_WEIGHT_RESUME_MATCH,
            hiring_probability=settings.RANKING_WEIGHT_HIRING_PROBABILITY,
            salary=settings.RANKING_WEIGHT_SALARY,
            recency=settings.RANKING_WEIGHT_RECENCY,
            company_quality=settings.RANKING_WEIGHT_COMPANY_QUALITY,
            competition=settings.RANKING_WEIGHT_COMPETITION,
        )

        results: dict[uuid.UUID, ranking.RankingBreakdown] = {}
        for match in matches:
            job = jobs_by_id[match.job_id]
            days_since_posted = (
                (now - job.posted_at).total_seconds() / 86400.0
                if job.posted_at is not None
                else float(decay_days)
            )
            hiring_probability = heuristics.compute_hiring_probability(
                hiring_frequency_per_month=hiring_freq_by_company.get(job.company_id),
                days_since_posted=days_since_posted,
                recency_decay_days=decay_days,
            )
            recency = ranking.compute_recency_score(
                days_since_posted=days_since_posted, recency_decay_days=decay_days
            )
            salary_score = ranking.compute_salary_percentile(
                float(job.salary_max) if job.salary_max is not None else None,
                comparison_salaries=comparison_salaries,
            )
            company_quality = ranking.compute_company_quality(
                company_rating=company_rating_by_company.get(job.company_id),
                avg_job_quality_score=avg_job_quality_by_company.get(job.company_id),
            )
            competition = ranking.compute_competition_score(
                strong_match_count=competition_by_job.get(job.id, 0)
            )
            resume_match = (
                float(match.match_score) if match.match_score is not None else None
            )
            results[job.id] = ranking.compute_ranking_score(
                resume_match=resume_match,
                hiring_probability=hiring_probability,
                salary=salary_score,
                recency=recency,
                company_quality=company_quality,
                competition=competition,
                weights=weights,
            )
        return results

    async def _feed_saved(
        self, user_id: uuid.UUID, pagination: PaginationParams
    ) -> tuple[list[JobFeedItem], int]:
        saved, total = await self._saved_job_repository.list_for_user(
            user_id=user_id, offset=pagination.offset, limit=pagination.page_size
        )
        jobs_by_id = await self._search_repository.fetch_jobs_by_ids([s.job_id for s in saved])
        pairs = [
            (self._search_service.to_result(jobs_by_id[s.job_id], None), None)
            for s in saved
            if s.job_id in jobs_by_id
        ]
        items = await self._to_feed_items(user_id, pairs)
        return items, total

    async def _feed_applied(
        self, user_id: uuid.UUID, pagination: PaginationParams
    ) -> tuple[list[JobFeedItem], int]:
        applications, total = await self._application_repository.list_applied_with_job(
            user_id=user_id, offset=pagination.offset, limit=pagination.page_size
        )
        job_ids = [a.job_id for a in applications if a.job_id is not None]
        jobs_by_id = await self._search_repository.fetch_jobs_by_ids(job_ids)
        pairs = [
            (self._search_service.to_result(jobs_by_id[a.job_id], None), None)
            for a in applications
            if a.job_id in jobs_by_id
        ]
        items = await self._to_feed_items(user_id, pairs)
        return items, total

    async def _feed_trending(
        self, user_id: uuid.UUID, pagination: PaginationParams
    ) -> tuple[list[JobFeedItem], int]:
        """Delegates to the existing `SearchRepository.trending_jobs` query
        (window + hiring-frequency ordering already implemented for the
        public `/search/trending/jobs` endpoint) rather than routing
        through `SearchService.search`, which has no "trending" sort mode.
        Pagination here is best-effort: `trending_jobs` takes a single
        `limit`, not a `(offset, limit)` window, matching its existing
        public-endpoint contract — this feed mode doesn't claim a precise
        total beyond what one bounded query returns.
        """
        fetch_limit = pagination.offset + pagination.page_size
        jobs = await self._search_repository.trending_jobs(
            window_days=self._settings.SEARCH_TRENDING_WINDOW_DAYS, limit=fetch_limit
        )
        total = len(jobs)
        page = jobs[pagination.offset : pagination.offset + pagination.page_size]
        pairs = [(self._search_service.to_result(job, None), None) for job in page]
        items = await self._to_feed_items(user_id, pairs)
        return items, total

    async def _feed_via_search(
        self,
        user_id: uuid.UUID,
        mode: JobFeedMode,
        filters: JobFeedFilters,
        pagination: PaginationParams,
    ) -> tuple[list[JobFeedItem], int]:
        """`all`/`newest` (and the zero-matches `recommended` fallback):
        delegates straight to the existing `SearchService.search`, reusing
        its filter/keyword/semantic/cache machinery unchanged.
        """
        base_filters = SearchFilters(**filters.model_dump(exclude={"feed_mode"}))
        if mode == JobFeedMode.NEWEST:
            base_filters.sort_by = SearchSortBy.NEWEST
            base_filters.sort_direction = SortDirection.DESC
        result_items, total = await self._search_service.search(
            filters=base_filters, pagination=pagination
        )
        pairs = [(item, None) for item in result_items]
        items = await self._to_feed_items(user_id, pairs)
        return items, total

    async def _to_feed_items(
        self, user_id: uuid.UUID | None, pairs: list[tuple]
    ) -> list[JobFeedItem]:
        """Batch-enrich a page of `(SearchResultItem, ResumeJobMatch | None)`
        pairs into `JobFeedItem`s — one query per lookup *for the whole
        page*, not per card, matching `SearchRepository.fetch_jobs_by_ids`'s
        existing batching convention. `user_id=None` (anonymous viewer of
        `GET /search/jobs/{job_id}`) skips every personalized lookup rather
        than querying with a placeholder id.
        """
        if not pairs:
            return []

        job_ids = [item.id for item, _ in pairs]
        company_ids = [item.company.id for item, _ in pairs]

        if user_id is not None:
            is_saved_ids = await self._saved_job_repository.list_job_ids_for_user(
                user_id=user_id, job_ids=job_ids
            )
            application_status_by_job = await self._application_repository.get_status_by_job_ids(
                user_id=user_id, job_ids=job_ids
            )
            current_resume_version = (
                await self._matching_repository.get_current_resume_version_for_user(
                    user_id=user_id
                )
            )
            resume_ats_baseline = (
                current_resume_version.ats_baseline_score
                if current_resume_version is not None
                else None
            )
        else:
            is_saved_ids = set()
            application_status_by_job = {}
            resume_ats_baseline = None

        hiring_freq_by_company = await self._search_repository.get_hiring_frequency_by_company_ids(
            company_ids
        )

        now = datetime.now(UTC)
        decay_days = self._settings.SEARCH_HIRING_PROB_RECENCY_DECAY_DAYS
        skill_threshold = self._settings.SEARCH_DIFFICULTY_SKILL_THRESHOLD

        items: list[JobFeedItem] = []
        for item, match in pairs:
            days_since_posted = (
                (now - item.posted_at).total_seconds() / 86400.0
                if item.posted_at is not None
                else float(decay_days)
            )
            hiring_probability = heuristics.compute_hiring_probability(
                hiring_frequency_per_month=hiring_freq_by_company.get(item.company.id),
                days_since_posted=days_since_posted,
                recency_decay_days=decay_days,
            )
            difficulty_label, difficulty_score = heuristics.compute_application_difficulty(
                required_skill_count=len(item.skills),
                experience_level_title=item.title,
                description_length=len(item.description),
                skill_count_threshold=skill_threshold,
            )

            match_context: JobMatchContext | None = None
            if match is not None:
                missing_count = len(match.missing_skills)
                total_count = len(match.matched_skills) + len(match.missing_skills)
                match_context = JobMatchContext(
                    match_score=float(match.match_score) if match.match_score is not None else None,
                    matched_skills=list(match.matched_skills),
                    missing_skills=list(match.missing_skills),
                    strengths=list(match.strengths),
                    weaknesses=list(match.weaknesses),
                    score_breakdown=dict(match.score_breakdown),
                )
            else:
                missing_count = 0
                total_count = len(item.skills)

            expected_ats_score = heuristics.estimate_ats_score(
                resume_ats_baseline_score=resume_ats_baseline,
                missing_skill_count=missing_count,
                total_skill_count=total_count,
            )

            application_status = application_status_by_job.get(item.id)
            items.append(
                JobFeedItem(
                    **item.model_dump(),
                    match=match_context,
                    ai_signals=JobAISignals(
                        hiring_probability=hiring_probability,
                        application_difficulty=difficulty_label,
                        application_difficulty_score=difficulty_score,
                        expected_ats_score=expected_ats_score,
                    ),
                    is_saved=item.id in is_saved_ids,
                    # `Application.status` is a `String(20)` column (not a
                    # DB-native `Enum`) — a raw, non-ORM-instance column
                    # select (see `ApplicationRepository.get_status_by_job_ids`)
                    # returns the plain string value, not an `ApplicationStatus`
                    # member, so this is passed through as-is (`str(...)`
                    # only to satisfy the type checker; it's already a str).
                    application_status=(
                        str(application_status) if application_status is not None else None
                    ),
                )
            )
        return items

    # --- single-job reads --------------------------------------------------------------

    async def get_job(
        self, *, user_id: uuid.UUID | None, job_id: uuid.UUID
    ) -> JobFeedItem:
        """Backs `GET /search/jobs/{job_id}` — works for anonymous viewing
        too (`user_id` may be `None`): saved/applied/match context are all
        blank for an anonymous caller rather than raising.
        """
        jobs_by_id = await self._search_repository.fetch_jobs_by_ids([job_id])
        job = jobs_by_id.get(job_id)
        if job is None:
            raise NotFoundError("Job not found.")

        result_item = self._search_service.to_result(job, None)
        match = None
        if user_id is not None:
            match = await self._matching_repository.get_match_by_user_and_job(
                user_id=user_id, job_id=job_id
            )
        items = await self._to_feed_items(user_id, [(result_item, match)])
        return items[0]

    async def get_why_score(
        self, *, user_id: uuid.UUID, job_id: uuid.UUID
    ) -> WhyThisScorePanel:
        jobs_by_id = await self._search_repository.fetch_jobs_by_ids([job_id])
        if job_id not in jobs_by_id:
            raise NotFoundError("Job not found.")
        job = jobs_by_id[job_id]

        match = await self._matching_repository.get_match_by_user_and_job(
            user_id=user_id, job_id=job_id
        )
        if match is None:
            return WhyThisScorePanel(
                matched_skills=[],
                missing_skills=[],
                skill_checklist=[],
                experience_match=None,
                location_match=None,
                salary_match=None,
                resume_suggestions=[],
                narrative=None,
            )

        breakdown = match.score_breakdown or {}
        boosts = breakdown.get("boosts", {})
        resume_suggestions = [
            f"Add {skill} to your resume to close this gap."
            for skill in list(match.missing_skills)[:_RESUME_SUGGESTION_LIMIT]
        ]
        skill_checklist = [
            SkillCheckItem(skill=skill, matched=matched)
            for skill, matched in build_skill_checklist(
                job_skills=[s.name for s in job.skills],
                matched_skills=list(match.matched_skills),
                missing_skills=list(match.missing_skills),
            )
        ]
        narrative = await self._get_or_generate_narrative(match)
        return WhyThisScorePanel(
            matched_skills=list(match.matched_skills),
            missing_skills=list(match.missing_skills),
            skill_checklist=skill_checklist,
            experience_match=_describe_boost(boosts.get("experience")),
            location_match=_describe_boost(boosts.get("country")),
            salary_match=_describe_boost(boosts.get("salary")),
            resume_suggestions=resume_suggestions,
            narrative=narrative,
        )

    async def _get_or_generate_narrative(self, match) -> str | None:  # noqa: ANN001
        """Best-effort one-line "why this fits" summary, cached on the match
        row. Never raises — an AI-provider hiccup should degrade to "no
        narrative," not break the whole why-score panel.
        """
        if match.ai_narrative:
            return match.ai_narrative
        try:
            from app.dependency_injection.container import get_container
            from app.modules.matching.narrative import generate_match_narrative

            ai_service = get_container().resolve("ai_service")
            narrative = await generate_match_narrative(
                ai_service=ai_service,
                matched_skills=list(match.matched_skills),
                missing_skills=list(match.missing_skills),
                strengths=list(match.strengths or []),
                weaknesses=list(match.weaknesses or []),
                match_score=match.match_score,
            )
        except Exception:
            logger.warning(
                "Match narrative generation failed, continuing without it",
                extra={"extra_fields": {"match_id": str(match.id)}},
                exc_info=True,
            )
            return None

        if narrative:
            match.ai_narrative = narrative
            await self._matching_repository.session.commit()
        return narrative

    # --- home summary --------------------------------------------------------------------

    async def get_home_summary(self, *, user_id: uuid.UUID) -> JobSearchHomeSummary:
        """Composed from existing repos exactly like `WorkspaceService.get_summary`.

        `recommended_action` is fetched by calling `WorkspaceService.get_summary`
        (via `workspace.router.get_workspace_service`, the exact same
        composition FastAPI's own `/workspace` endpoint uses) and taking its
        first next-best-action — not reimplementing that prioritization
        logic here. `applications_needing_follow_up_count` is read off the
        same `workspace_summary.applications.needs_follow_up` list
        `TrackingService.get_dashboard` already computes, for the same
        reason.
        """
        from app.modules.workspace.router import get_workspace_service

        now = datetime.now(UTC)
        since_yesterday = now - timedelta(days=1)
        trending_window_start = now - timedelta(days=self._settings.SEARCH_TRENDING_WINDOW_DAYS)

        matches = await self._matching_repository.list_matches_for_user(
            user_id=user_id, limit=self._settings.MATCHING_CANDIDATE_LIMIT
        )
        scores = [float(m.match_score) for m in matches if m.match_score is not None]
        average_match_pct = round(sum(scores) / len(scores), 2) if scores else None
        highest_match_pct = round(max(scores), 2) if scores else None
        strong_match_count = sum(1 for s in scores if s >= 80.0)

        top_missing_skill: str | None = None
        top_missing_skill_count = 0
        missing_skill_counts: dict[str, int] = {}
        for m in matches:
            for skill in m.missing_skills:
                missing_skill_counts[skill] = missing_skill_counts.get(skill, 0) + 1
        if missing_skill_counts:
            top_missing_skill, top_missing_skill_count = max(
                missing_skill_counts.items(), key=lambda pair: pair[1]
            )

        easy_apply_count = 0
        if matches:
            jobs_by_id = await self._search_repository.fetch_jobs_by_ids(
                [m.job_id for m in matches]
            )
            skill_threshold = self._settings.SEARCH_DIFFICULTY_SKILL_THRESHOLD
            for m in matches:
                job = jobs_by_id.get(m.job_id)
                if job is None:
                    continue
                difficulty_label, _ = heuristics.compute_application_difficulty(
                    required_skill_count=len(job.skills),
                    experience_level_title=job.title,
                    description_length=len(job.description),
                    skill_count_threshold=skill_threshold,
                )
                if difficulty_label == "easy":
                    easy_apply_count += 1

        new_jobs_since_yesterday = await self._search_repository.count_jobs_posted_since(
            since=since_yesterday
        )
        companies_hiring_now = await self._search_repository.count_distinct_hiring_companies(
            since=trending_window_start
        )

        workspace_service = get_workspace_service(self._session)
        workspace_summary = await workspace_service.get_summary(user_id=user_id)
        recommended_action = (
            workspace_summary.next_best_actions[0]
            if workspace_summary.next_best_actions
            else None
        )
        applications_needing_follow_up_count = len(
            workspace_summary.applications.needs_follow_up
        )

        return JobSearchHomeSummary(
            greeting=_build_greeting(now),
            jobs_matching_profile_count=len(matches),
            new_jobs_since_yesterday=new_jobs_since_yesterday,
            average_match_pct=average_match_pct,
            highest_match_pct=highest_match_pct,
            companies_hiring_now=companies_hiring_now,
            recommended_action=recommended_action,
            strong_match_count=strong_match_count,
            top_missing_skill=top_missing_skill,
            top_missing_skill_count=top_missing_skill_count,
            easy_apply_count=easy_apply_count,
            applications_needing_follow_up_count=applications_needing_follow_up_count,
        )


__all__ = ["JobFeedService", "SavedJobService"]
